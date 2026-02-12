"""
Evaluate an unwrapped autoencoder model using ViSQOL.

Usage:
    python evaluate_autoencoder.py \
        --model_config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json \
        --ckpt_path exported_model.ckpt \
        --dataset_config stable_audio_tools/configs/dataset_configs/local_dac.json \
        --batch_size 16 \
        --num_workers 6 \
        --max_samples 0          # 0 = use all samples
        --device cuda

    You can also use a YAML config file:
        python evaluate_autoencoder.py --args.load config.yml

Requires:
    pip install visqol argbind
"""

import argbind
import json
import os
import sys

import numpy as np
import scipy.stats
import torch
import torchaudio.transforms as T
from tqdm import tqdm
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2, similarity_result_pb2

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict


# ---------------------------------------------------------------------------
# ViSQOL helpers
# ---------------------------------------------------------------------------

VISQOL_SAMPLE_RATE = 48000  # ViSQOL audio mode requires 48 kHz


def make_visqol_api(use_speech_mode: bool = False):
    """Create and return a VisqolApi instance."""
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = VISQOL_SAMPLE_RATE
    config.options.use_speech_scoring = use_speech_mode

    if use_speech_mode:
        svr_model_name = "lattice_tcditool_in_speech.tflite"
    else:
        svr_model_name = "libsvm_nu_svr_model.txt"

    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_name
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    return api


def compute_visqol(api, reference: np.ndarray, degraded: np.ndarray) -> float:
    """
    Compute ViSQOL MOS-LQO between a reference and degraded signal.

    Both inputs should be 1-D float64 numpy arrays at 48 kHz.
    Returns the MOS-LQO score.
    """
    similarity_result = api.Measure(reference, degraded)
    return similarity_result.moslqo


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------

def mean_confidence_interval(data, confidence=0.95):
    """Return (mean, ci_margin) for *data* at the given confidence level."""
    a = np.array(data, dtype=np.float64)
    n = len(a)
    mean = np.mean(a)
    if n < 2:
        return mean, 0.0
    return mean, scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@argbind.bind(without_prefix=True)
def evaluate(
    model_config: str = "",
    ckpt_path: str = "",
    dataset_config: str = "",
    batch_size: int = 16,
    num_workers: int = 6,
    max_samples: int = 0,
    device: str = "cuda",
    speech_mode: bool = False,
):
    """Evaluate an unwrapped autoencoder using ViSQOL (encode → decode → compare).

    Args:
        model_config: Path to the model config JSON (e.g. stable_audio_1_0_vae.json).
        ckpt_path: Path to the unwrapped model checkpoint (.ckpt or .safetensors).
        dataset_config: Path to the dataset config JSON.
        batch_size: Batch size for inference.
        num_workers: Number of dataloader workers.
        max_samples: Maximum number of samples to evaluate. 0 = all.
        device: Device to run the model on.
        speech_mode: Use ViSQOL speech mode instead of audio mode.
    """
    assert model_config, "--model_config is required"
    assert ckpt_path, "--ckpt_path is required"
    assert dataset_config, "--dataset_config is required"

    # ---- load configs ----
    with open(model_config) as f:
        model_cfg = json.load(f)

    with open(dataset_config) as f:
        dataset_cfg = json.load(f)
    dataset_cfg["random_crop"] = False
    dataset_cfg["drop_last"] = False

    # ---- build model and load weights ----
    print("Creating model from config …")
    model = create_model_from_config(model_cfg)
    print(f"Loading checkpoint from {ckpt_path} …")
    copy_state_dict(model, load_ckpt_state_dict(ckpt_path))
    model.to(device).eval().requires_grad_(False)
    print("Model ready.")

    # ---- build dataloader ----
    data_loader = create_dataloader_from_config(
        dataset_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=model_cfg["sample_rate"],
        sample_size=model_cfg["sample_size"],
        audio_channels=model_cfg.get("audio_channels", 1),
        shuffle=False,
    )

    # ---- set up ViSQOL ----
    print("Initialising ViSQOL …")
    visqol_api = make_visqol_api(use_speech_mode=speech_mode)

    # Resampler from model sample rate → 48 kHz (if needed)
    if model_cfg["sample_rate"] != VISQOL_SAMPLE_RATE:
        resampler = T.Resample(model_cfg["sample_rate"], VISQOL_SAMPLE_RATE)
    else:
        resampler = None

    # ---- evaluation loop ----
    scores = []
    total_processed = 0
    effective_max = max_samples if max_samples > 0 else float("inf")

    pbar = tqdm(data_loader, desc="Evaluating", unit="batch")
    for batch in pbar:
        audio = batch[0].to(device)  # (B, C, T)

        with torch.no_grad():
            reconstructed = model.decode(model.encode(audio))

        # Ensure reconstructed has the same length as original
        min_len = min(audio.shape[-1], reconstructed.shape[-1])
        # Move to CPU for ViSQOL computation
        audio_cpu = audio[..., :min_len].cpu()
        reconstructed_cpu = reconstructed[..., :min_len].cpu()

        for i in range(audio_cpu.shape[0]):
            if total_processed >= effective_max:
                break

            # Mix down to mono (ViSQOL expects mono)
            ref = audio_cpu[i].mean(dim=0)          # (T,)
            deg = reconstructed_cpu[i].mean(dim=0)   # (T,)

            # Resample to 48 kHz if needed
            if resampler is not None:
                ref = resampler(ref.unsqueeze(0)).squeeze(0)
                deg = resampler(deg.unsqueeze(0)).squeeze(0)

            # Convert to float64 numpy (ViSQOL requirement)
            ref_np = ref.numpy().astype(np.float64)
            deg_np = deg.numpy().astype(np.float64)

            try:
                score = compute_visqol(visqol_api, ref_np, deg_np)
                scores.append(score)
                total_processed += 1
            except Exception as e:
                print(f"\n[WARNING] ViSQOL failed on sample {total_processed}: {e}", file=sys.stderr)

            # Update progress bar with running statistics
            if scores:
                running_mean = np.mean(scores)
                pbar.set_postfix(visqol=f"{running_mean:.4f}", n=len(scores))

        if total_processed >= effective_max:
            break

    # ---- report results ----
    if not scores:
        print("\nNo scores were computed. Check your dataset / model.")
        return

    mean, ci = mean_confidence_interval(scores, confidence=0.95)

    print("\n" + "=" * 60)
    print(f"  ViSQOL Evaluation Results  ({len(scores)} samples)")
    print("=" * 60)
    print(f"  Mean MOS-LQO :  {mean:.4f} ± {ci:.4f}")
    print(f"  Std Dev      :  {np.std(scores):.4f}")
    print(f"  Min / Max    :  {np.min(scores):.4f} / {np.max(scores):.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
