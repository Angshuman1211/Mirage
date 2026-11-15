#!/bin/bash
# H8 - build a WORKING petitRADTRANS in its own env (`mirage-prt`).
#
# Why a separate env, and why it has to be rebuilt at all:
# the pRT 3.3.3 already present in `mirage` imports but cannot run. Every one of
# its five f2py extensions has an UNDEFINED _f2pyinit<module>_ symbol, because
# the only gfortran on this machine is x86_64 (/usr/local/bin/gfortran, target
# x86_64-apple-darwin24, from the Intel Homebrew prefix) while the Mac and its
# Python are arm64. The Fortran objects were built for the wrong architecture
# and the arm64 link dropped them, so the C halves loaded and the Fortran halves
# were simply absent. pRT ships source-only on PyPI and is not on conda-forge,
# so there is no prebuilt binary to fall back on: it must be compiled by an
# arm64 Fortran compiler, which conda-forge provides.
#
# Separate env follows the existing `mirage-taurex` split: H8 compares two codes
# that cannot share one dependency set, so each runs in its own and they meet
# through .npz files on disk.
set -euo pipefail
CONDA=/Users/angshumansmac/anaconda3
ENVDIR=$CONDA/envs/mirage-prt

if [ ! -x "$ENVDIR/bin/python" ]; then
  echo "[prt-env] creating mirage-prt  $(date +%H:%M:%S)"
  "$CONDA/bin/conda" create -y -n mirage-prt -c conda-forge \
      python=3.11 gfortran numpy scipy h5py astropy
else
  echo "[prt-env] env exists, skipping create"
fi

echo "[prt-env] fortran compiler in env:"
ls "$ENVDIR"/bin/*gfortran* 2>/dev/null || { echo "  NO gfortran in env"; exit 1; }
"$ENVDIR"/bin/gfortran -dumpmachine

# pure-python pRT deps that matter for a forward model. Deliberately NOT
# installing pymultinest / mpi4py / pyvo / skycalc-cli: they are for retrievals,
# MPI and sky models, none of which H8 uses, and mpi4py needs an MPI toolchain
# that is not here. selenium and bs4 ARE needed despite H8 never scraping
# anything: prt_cli imports them at module level and sits in the Radtrans
# import chain, so without them `from petitRADTRANS.radtrans import Radtrans`
# dies on ModuleNotFoundError.
echo "[prt-env] python deps"
"$ENVDIR"/bin/pip install -q molmass dill requests selenium bs4

# Build deps go in the env, and the build runs with --no-build-isolation.
# pip's isolated build subprocess could not reach the package index here
# ("Could not find a version that satisfies meson-python ... from versions: none")
# even though the outer pip downloads fine, so isolation is the thing to drop.
echo "[prt-env] build toolchain"
"$ENVDIR"/bin/pip install -q "meson-python>=0.12.1" meson ninja

echo "[prt-env] building petitRADTRANS 3.3.3 from source  $(date +%H:%M:%S)"
export PATH="$ENVDIR/bin:$PATH"
export FC="$ENVDIR/bin/gfortran"
export F90="$FC"
"$ENVDIR"/bin/pip install --no-deps --no-build-isolation --no-cache-dir --force-reinstall \
    "petitRADTRANS==3.3.3" 2>&1 | tail -20

echo "[prt-env] verifying the Fortran halves actually linked"
SO=$("$ENVDIR"/bin/python -c "import os,petitRADTRANS as p;print(os.path.dirname(p.__file__))" 2>/dev/null || echo "")
bad=0
for f in "$SO"/*.so; do
  if nm "$f" | grep -q "U _f2pyinit"; then echo "  UNDEFINED still in $(basename "$f")"; bad=1
  else echo "  ok $(basename "$f")"; fi
done
[ "$bad" -eq 0 ] || { echo "[prt-env] FAILED: f2py init symbols still undefined"; exit 1; }
"$ENVDIR"/bin/python -c "
from petitRADTRANS.radtrans import Radtrans
print('[prt-env] Radtrans imports OK')
"
echo "[prt-env] done  $(date +%H:%M:%S)"
