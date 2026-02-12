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
SYSTEM_SO=$(ls /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.* 2>/dev/null | sort -V | tail -1)
CONDA_SO=$(ls "${CONDA_LIB}"/libstdc++.so.6.0.* 2>/dev/null | sort -V | tail -1)
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

# ---- Install the project and all dependencies (including visqol) ----
echo "Installing stable-audio-tools and dependencies ..."
pip install -e .

echo ""
echo "Installation complete."

