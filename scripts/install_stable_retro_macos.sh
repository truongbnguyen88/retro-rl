#!/usr/bin/env bash
# Build + install stable-retro from source on macOS, patching the vendored
# zlib copies that break against macOS 11+ SDKs.
#
# Why this script exists
# ----------------------
# stable-retro's PyPI arm64 wheels (1.0.0, 0.9.9) are mislabeled — they tag
# arm64 but contain x86_64 binaries, so `dlopen` fails on Apple Silicon.
# Source build fails because the libretro cores ship multiple copies of an
# ancient zlib whose `zutil.h` does `#define fdopen(fd,mode) NULL` on any
# platform where `TARGET_OS_MAC` is set — that includes modern Darwin. When
# the system `<stdio.h>` then declares `FILE *fdopen(int, const char *)`,
# the preprocessor substitutes `NULL` and the SDK header fails to parse.
#
# This script patches every vendored `zutil.h` to also skip the bad redef
# on `__APPLE__`, then builds + installs from a git clone.
#
# Prereqs
# -------
#   brew install cmake pkg-config
#   active python venv with `pip` (`source .venv/bin/activate`)
#
# Usage
# -----
#   ./scripts/install_stable_retro_macos.sh

set -euo pipefail

if ! command -v cmake >/dev/null || ! command -v pkg-config >/dev/null; then
  echo "ERROR: cmake and pkg-config are required. Install with:"
  echo "       brew install cmake pkg-config"
  exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: activate a python venv first (source .venv/bin/activate)"
  exit 1
fi

WORK_DIR=$(mktemp -d -t stable-retro-build-XXXXXX)
trap 'rm -rf "$WORK_DIR"' EXIT

echo ">> cloning stable-retro into $WORK_DIR"
git clone --recursive --depth 1 \
  https://github.com/Farama-Foundation/stable-retro.git "$WORK_DIR/stable-retro"

cd "$WORK_DIR/stable-retro"

echo ">> patching vendored zlib copies (fdopen redef guard)"
patched=0
for f in $(find . -name "zutil.h"); do
  # Two indent variants are present in different vendored copies.
  if grep -qE '^#( {2,6})ifndef fdopen$' "$f" && ! grep -q '!defined(__APPLE__)' "$f"; then
    sed -i.bak -E 's|^#(\s+)ifndef fdopen$|#\1if !defined(fdopen) \&\& !defined(__APPLE__)|' "$f"
    patched=$((patched + 1))
  fi
done
echo ">> patched $patched zutil.h files"

echo ">> running pip install (this will trigger cmake + make)"
pip install . --no-build-isolation

echo ">> verifying install"
python -c "import retro; retro.data.get_romfile_path('Airstriker-Genesis-v0'); print('OK: stable-retro imports + Airstriker ROM resolvable')"
