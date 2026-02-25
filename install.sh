#!/usr/bin/env bash
set -euo pipefail

# ---- Install Bazel via Bazelisk (if not already installed) ----
if ! command -v bazel &> /dev/null; then
    echo "Installing Bazelisk (Bazel launcher) ..."
    if command -v npm &> /dev/null; then
        npm install -g @bazel/bazelisk
    else
        # Fallback: download bazelisk binary directly
        BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
        python -c "import urllib.request; urllib.request.urlretrieve('$BAZELISK_URL', '/usr/local/bin/bazel')"
        chmod +x /usr/local/bin/bazel
    fi
    echo "Bazel installed: $(bazel --version)"
else
    echo "Bazel already installed: $(bazel --version)"
fi

# ---- Fix conda's outdated libstdc++ if the system has a newer one ----
CONDA_LIB="${CONDA_PREFIX:-/opt/conda}/lib"
SYSTEM_SO=$(ls /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.* 2>/dev/null | sort -V | tail -1 || true)
CONDA_SO=$(ls "${CONDA_LIB}"/libstdc++.so.6.0.* 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$SYSTEM_SO" ] && [ -n "$CONDA_SO" ]; then
    SYSTEM_VER=$(basename "$SYSTEM_SO")
    CONDA_VER=$(basename "$CONDA_SO")
    if [ "$(printf '%s\n' "$CONDA_VER" "$SYSTEM_VER" | sort -V | tail -1)" != "$CONDA_VER" ]; then
        echo "Updating conda libstdc++ (${CONDA_VER}) with system version (${SYSTEM_VER}) ..."
        cp "$SYSTEM_SO" "${CONDA_LIB}/${SYSTEM_VER}"
        ln -sf "$SYSTEM_VER" "${CONDA_LIB}/libstdc++.so.6"
        ln -sf "$SYSTEM_VER" "${CONDA_LIB}/libstdc++.so"
    fi
fi

# ---- Ensure git uses HTTPS (not SSH) so .netrc credentials work ----
export HOME=/data/alejandro
export GIT_CONFIG_GLOBAL=/dev/null

# ---- Build & install visqol from source (needs Bazel + GCC ≥13 patch) ----
echo "Building visqol from source ..."
VISQOL_TMP=$(mktemp -d)
git clone --depth 1 https://github.com/google/visqol.git "$VISQOL_TMP"
pushd "$VISQOL_TMP" > /dev/null

bazel build -c opt //:similarity_result_py_pb2 //:visqol_config_py_pb2

# Patch TF-Lite spectrogram.cc for GCC 13+ (missing <cstdint>)
for f in $(find "${HOME}/.cache/bazel" -name "spectrogram.cc" -path "*/tensorflow/lite/*" 2>/dev/null); do
    grep -q '<cstdint>' "$f" || sed -i 's|#include <math.h>|#include <math.h>\n#include <cstdint>|' "$f"
done

bazel build -c opt //python:visqol_lib_py.so
pip install .

popd > /dev/null
rm -rf "$VISQOL_TMP"
echo "visqol installed."

# ---- Install the project and remaining dependencies ----
echo "Installing stable-audio-tools and dependencies ..."
pip install -e .

echo ""
echo "Installation complete."

