#!/usr/bin/env bash
# Install `llama-server` (llama.cpp) OUTSIDE the repo, for the fmd pipeline.
#
# Decision of record: docs/product-development-cycle/10-inference-and-model-selection.md
# ("Decision update — 2026-11-17 (locked: Gemma 4 26B-A4B / UD-IQ4_XS)") and
# 11-inference-runbook.md §2.7. We serve the GGUF stack with llama.cpp, not
# vLLM.
#
# This host: RTX 5080 Laptop, sm_120 (Blackwell), 16 GB, driver 580.178.04
# (max CUDA 13.0), CUDA toolkit 13.0.88 at /usr/local/cuda, 24 cores.
#
# WHY A SOURCE BUILD IS THE DEFAULT HERE (measured 2026-11-17):
#   The prebuilt `llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz` release asset
#   links `libcudart.so.12`, and this host only ships the CUDA 13 runtime
#   (`libcudart.so.13`). Making the prebuilt run would mean pulling the extra
#   594 MB `cudart-llama-...-cuda-12.8-x64` asset. A source build against the
#   installed toolkit produces the same code with no ABI shim, natively for
#   sm_120. Cost: ~6 min on 12 jobs (measured).
#
#   scripts/install_llama_cpp.sh            # build + install (default)
#   scripts/install_llama_cpp.sh --verify   # re-check version + CUDA devices
#   scripts/install_llama_cpp.sh --prebuilt # release tarball instead (needs cudart)
#
# Environment:
#   LLAMA_ROOT       install root        (default ~/.local/llama.cpp)
#   LLAMA_TAG        release tag         (default b12468 — first tag after the
#                                        Gemma 4 MTP merge, PR #23398, 2026-06-07)
#   LLAMA_CUDA_TAG   prebuilt asset tag  (default cuda-12.8-x64)
#   CUDA_ARCH        CMAKE_CUDA_ARCHITECTURES (default 120 = this GPU)
#   LLAMA_JOBS       build parallelism   (default min(nproc,12); higher OOMs
#                                        this host's 30 GB of RAM)
#
# Nothing here is baked into an image and nothing lands in the repo tree.

set -euo pipefail

ROOT="${LLAMA_ROOT:-$HOME/.local/llama.cpp}"
TAG="${LLAMA_TAG:-b12468}"
CUDA_TAG="${LLAMA_CUDA_TAG:-cuda-12.8-x64}"
ARCH="${CUDA_ARCH:-120}"
JOBS="${LLAMA_JOBS:-$(nproc)}"
[ "$JOBS" -gt 12 ] && JOBS=12
SRC="$ROOT/src"
BIN_LINK="$ROOT/bin"

die() { echo "error: $*" >&2; exit 1; }

cmd_build() {
  [ -x "$(command -v nvcc)" ] || die "nvcc not on PATH; a CUDA build needs the toolkit (or use --prebuilt)"
  [ -x "$(command -v cmake)" ] || die "cmake not installed"

  if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$ROOT"
    echo "cloning llama.cpp $TAG into $SRC"
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
  fi

  cmake -S "$SRC" -B "$SRC/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF >"$ROOT/build-configure.log" 2>&1 \
    || { tail -20 "$ROOT/build-configure.log"; die "cmake configure failed"; }

  echo "building llama-server (arch sm_$ARCH, -j $JOBS) — ~6 min"
  cmake --build "$SRC/build" --config Release -j "$JOBS" \
    --target llama-server llama-cli >"$ROOT/build.log" 2>&1 \
    || { tail -30 "$ROOT/build.log"; die "build failed"; }

  mkdir -p "$BIN_LINK"
  # Symlink rather than copy: the binaries need the .so files beside them.
  for f in "$SRC"/build/bin/*; do ln -sfn "$f" "$BIN_LINK/$(basename "$f")"; done
  echo "installed: $BIN_LINK/llama-server -> $SRC/build/bin/llama-server"
  cmd_verify
}

cmd_prebuilt() {
  local arch="x64"
  [ "$(uname -m)" = "aarch64" ] && arch="arm64"
  local asset="llama-$TAG-bin-ubuntu-${CUDA_TAG}-${arch}.tar.gz"
  local url="https://github.com/ggml-org/llama.cpp/releases/download/$TAG/$asset"
  local dest="$ROOT/llama-$TAG"

  if [ -x "$dest/llama-server" ] || [ -x "$dest/bin/llama-server" ]; then
    echo "already installed: $dest"
    ln -sfn "$([ -d "$dest/bin" ] && echo "$dest/bin" || echo "$dest")" "$BIN_LINK"
    cmd_verify
    return 0
  fi

  mkdir -p "$ROOT"
  echo "downloading $asset"
  # -C -: the host's link is slow and drops; resume rather than restart.
  curl -4 -L --retry 8 --retry-delay 3 -C - -o "$ROOT/$asset" "$url"
  tar -xzf "$ROOT/$asset" -C "$ROOT"
  [ -d "$ROOT/build" ] && dest="$ROOT/build"
  ln -sfn "$dest/bin" "$BIN_LINK"
  cmd_verify
}

cmd_verify() {
  local srv="$BIN_LINK/llama-server"
  [ -x "$srv" ] || die "no llama-server under $ROOT — run $0"
  "$srv" --version 2>&1 | head -3
  echo "--- CUDA devices ---"
  "$srv" --list-devices 2>&1 | head -20 || true
  echo "--- shared library check ---"
  if ldd "$srv" 2>/dev/null | grep "not found"; then
    die "missing shared libs (see above) — for a prebuilt asset also extract the matching cudart tarball"
  else
    echo "all shared libraries resolve"
  fi
}

case "${1:-build}" in
  build|install) cmd_build ;;
  --verify|verify) cmd_verify ;;
  --prebuilt|prebuilt) cmd_prebuilt ;;
  *) echo "usage: $0 [build|--verify|--prebuilt]" >&2; exit 2 ;;
esac
