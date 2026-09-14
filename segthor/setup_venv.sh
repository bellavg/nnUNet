#!/bin/bash
# Build the venv nnU-Net trains from, directly on a Snellius login node --
# no SLURM job needed for this step. pip install needs network access, which
# login nodes have and compute nodes may not, and installing packages is
# light enough (mostly I/O) that it doesn't need a genoa/CPU allocation.
#
# Usage (from this nnUNet checkout root, e.g. ~/nnUNet):
#   bash segthor/setup_venv.sh
# Override the venv path:
#   VENV_PATH=/my/path bash segthor/setup_venv.sh
#
# This is the exact same install logic as segthor/slurm/rebuild_venv.job --
# use that instead if/when you have genoa access again and would rather not
# tie up your login-node shell while torch installs.

set -euo pipefail

# pip's build-isolation step uses $TMPDIR for its temp build envs. On Snellius
# login nodes this can default to a small per-node scratch pool with its own
# separate (and easily exhausted) quota, unrelated to $HOME or scratch-shared
# -- redirect it to $HOME, which has room for the ~100s of MB a build env needs.
export TMPDIR="${TMPDIR_OVERRIDE:-$HOME/.tmp_pip_build}"
mkdir -p "$TMPDIR"

VENV_PATH="${VENV_PATH:-$HOME/.venv_nnunet}"
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [ ! -f "$BASE_DIR/pyproject.toml" ] && [ ! -f "$BASE_DIR/setup.py" ]; then
    echo "ERROR: BASE_DIR=$BASE_DIR does not look like an nnUNet checkout" \
         "(no pyproject.toml/setup.py). Run this from the checkout root," \
         "or set BASE_DIR explicitly." >&2
    exit 1
fi

echo "Building venv at: $VENV_PATH"
echo "From nnUNet checkout at: $BASE_DIR"

export PIP_NO_CACHE_DIR=1

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

if [[ -d "$VENV_PATH" ]]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_PATH"
fi

python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel

echo "Installing nnUNet (editable) + its dependencies..."
pip install -e "$BASE_DIR"

echo "Verifying imports..."
python -c "
import torch, nnunetv2
print(f'torch={torch.__version__}, cuda available={torch.cuda.is_available()}')
print(f'nnunetv2 importable from {nnunetv2.__file__}')
print('Build successful.')
"
