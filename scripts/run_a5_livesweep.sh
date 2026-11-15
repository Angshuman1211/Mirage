#!/bin/bash
# A5 part 1 (G4): does the NS anchor move with live points?
#
# WASP-96b is the cheapest target and the only one whose radius 68% interval
# contains the literature value (P5-D7), so it is both the affordable sweep and
# the one place the anchor is currently load-bearing.
#
# The published WASP-96b anchor used 150 live points, NOT the 300 that G4 assumes
# (data/real_ess/ns_wasp96.log line 1). So the sweep is anchored at 150 and the
# published value is re-run under current code rather than compared across versions.
#
# Output naming keeps the legacy `wasp39b_ns_posterior_<tag>.npz` scheme on purpose:
# fig_multitarget.py, multitarget_table.py and real_compare.py all hardcode it, so
# renaming would break them. Tags disambiguate instead. The published run at
# wasp39b_ns_posterior_wasp96.npz is never touched.
set -u
cd "/Users/angshumansmac/Desktop/Actual Projects/Exoplanet/Project"
PY=/Users/angshumansmac/anaconda3/envs/mirage-taurex/bin/python
export PYTHONPATH=scripts
L=data/real_ess
OBS=$L/wasp96_samples.npz

run() {
  n=$1
  out="$L/wasp39b_ns_posterior_wasp96_live${n}.npz"
  if [ -f "$out" ]; then echo "[a5] SKIP live=$n, $out exists"; return; fi
  echo "[a5] start live=$n $(date '+%T')"
  OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 "$PY" scripts/taurex_retrieve.py \
      --planet wasp96 --fitrad --live "$n" --tag "wasp96_live${n}" --obs-npz "$OBS" \
      > "$L/ns_wasp96_live${n}.log" 2>&1
  if [ -f "$out" ]; then echo "[a5] DONE live=$n $(date '+%T')"; else echo "[a5] FAILED live=$n"; fi
}

echo "[a5] live-point sweep launching 4 parallel runs $(date '+%T')"
run 150 &
run 300 &
run 600 &
run 1000 &
wait
echo "[a5] sweep complete $(date '+%T')"
ls -l $L/wasp39b_ns_posterior_wasp96_live*.npz 2>/dev/null
