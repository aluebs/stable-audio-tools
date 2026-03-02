#!/bin/bash
set -e

CONFIG_DIR="stable_audio_tools/configs/model_configs/autoencoders"
DATASET_CONFIG="stable_audio_tools/configs/dataset_configs/local_daps_test.json"
BATCH_SIZE=8
MASTER_PORT=29502
OUT_PATH="output"

echo "=========================================="
echo "Running VAE 1 baseline"
echo "=========================================="
torchrun --nproc_per_node=gpu --master_port=${MASTER_PORT} evaluate_autoencoder.py \
    --model_config ${CONFIG_DIR}/stable_audio_1_0_vae.json \
    --ckpt_path vae1_baseline.ckpt \
    --dataset_config ${DATASET_CONFIG} \
    --out_path ${OUT_PATH}/vae1_baseline \
    --batch_size ${BATCH_SIZE}

echo "=========================================="
echo "Running VAE 1 powerchannel"
echo "=========================================="
torchrun --nproc_per_node=gpu --master_port=${MASTER_PORT} evaluate_autoencoder.py \
    --model_config ${CONFIG_DIR}/stable_audio_1_0_vae_powerchannel.json \
    --ckpt_path vae1_powerchannel.ckpt \
    --dataset_config ${DATASET_CONFIG} \
    --out_path ${OUT_PATH}/vae1_powerchannel \
    --batch_size ${BATCH_SIZE}

echo "=========================================="
echo "Running VAE 2 baseline"
echo "=========================================="
torchrun --nproc_per_node=gpu --master_port=${MASTER_PORT} evaluate_autoencoder.py \
    --model_config ${CONFIG_DIR}/stable_audio_2_0_vae.json \
    --ckpt_path vae2_baseline.ckpt \
    --dataset_config ${DATASET_CONFIG} \
    --out_path ${OUT_PATH}/vae2_baseline \
    --batch_size ${BATCH_SIZE}

echo "=========================================="
echo "Running VAE 2 powerchannel"
echo "=========================================="
torchrun --nproc_per_node=gpu --master_port=${MASTER_PORT} evaluate_autoencoder.py \
    --model_config ${CONFIG_DIR}/stable_audio_2_0_vae_powerchannel.json \
    --ckpt_path vae2_powerchannel.ckpt \
    --dataset_config ${DATASET_CONFIG} \
    --out_path ${OUT_PATH}/vae2_powerchannel \
    --batch_size ${BATCH_SIZE}

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
