#!/bin/bash
# H8 - fetch the petitRADTRANS opacity tables needed for the cross-code check.
#
# Only the 5 molecules MIRAGE/TauREx actually fit, plus H2-H2 / H2-He CIA. The
# line lists are chosen to MATCH data/opacity_hifi (ExoMolOP R=15000) exactly,
# so a pRT-vs-TauREx disagreement is attributable to radiative transfer and not
# to opacity data:
#   H2O POKAZATEL, CO2 UCL-4000, CH4 YT34to10, NH3 CoYuTe  - identical lists
#   CO  C-O-NatAbund__Chubb, the ExoMolOP compilation
#
# The CO choice is the one that is not obvious from the filenames. pRT also
# ships 12C-16O__HITEMP, whose name looks like the closer match to TauREx's
# 12C-16O__Li2015 (same isotopologue, and HITEMP's CO is often described as
# Li et al. 2015). It is not: that table carries DOI 10.1016/j.jqsrt.2010.05.001,
# which is HITEMP 2010, whose CO came from Goorvitch 1994. The Chubb table
# carries DOI 10.1088/0067-0049/216/1/15, byte-identical to the DOI stored in
# data/opacity_hifi/12C-16O__Li2015...h5. Matching was checked on the DOI inside
# each file, never on the filename. The cost is that Chubb is natural abundance
# while TauREx's is the main isotopologue alone.
#
# pRT's own downloader (petitRADTRANS.cli.prt_cli) drives Selenium + Chrome
# because the URL in its config file omits the /files/ path segment and so
# returns the Keeper web app instead of the file. The direct share-link
# endpoint needs no browser, so this fetches over plain HTTP instead.
set -u
PY=/Users/angshumansmac/anaconda3/envs/mirage/bin/python
ROOT=/Users/angshumansmac/petitRADTRANS/input_data
TOKEN=ccf25082fda448c8a0d0
BASE="https://keeper.mpdl.mpg.de/d/${TOKEN}/files/?p="

L=opacities/lines/correlated_k
C=opacities/continuum/collision_induced_absorptions
FILES=(
  "/$C/H2--H2/H2--H2-NatAbund/H2--H2-NatAbund__BoRi.R831_0.6-250mu.ciatable.petitRADTRANS.h5"
  "/$C/H2--He/H2--He-NatAbund/H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu.ciatable.petitRADTRANS.h5"
  "/$L/CO/C-O-NatAbund/C-O-NatAbund__Chubb.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
  "/$L/H2O/1H2-16O/1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
  "/$L/CO2/12C-16O2/12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
  "/$L/CH4/12C-1H4/12C-1H4__YT34to10.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
  "/$L/NH3/14N-1H3/14N-1H3__CoYuTe.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
)

ok=0; bad=0
for p in "${FILES[@]}"; do
  out="${ROOT}${p}"
  name=$(basename "$p")
  # already a valid HDF5 file? skip. (A failed run can leave an HTML stub.)
  if [ -f "$out" ] && $PY -c "import h5py,sys; h5py.File(sys.argv[1]).close()" "$out" 2>/dev/null; then
    echo "[prt] SKIP $name ($(du -h "$out" | cut -f1))"; ok=$((ok+1)); continue
  fi
  # a partial file is resumable, an HTML stub is not: drop anything too small
  if [ -f "$out" ] && [ "$(stat -f%z "$out")" -lt 100000 ]; then rm -f "$out"; fi
  mkdir -p "$(dirname "$out")"
  q=$($PY -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$p")
  echo "[prt] GET  $name  $(date +%H:%M:%S)"
  curl -sL --retry 3 --retry-delay 5 -C - --max-time 3600 -o "$out" "${BASE}${q}&dl=1"
  if $PY -c "import h5py,sys; h5py.File(sys.argv[1]).close()" "$out" 2>/dev/null; then
    echo "[prt] OK   $name  $(du -h "$out" | cut -f1)  $(date +%H:%M:%S)"; ok=$((ok+1))
  else
    echo "[prt] FAIL $name  (not HDF5, $(du -h "$out" 2>/dev/null | cut -f1))"; bad=$((bad+1))
  fi
done
echo "[prt] done: $ok ok, $bad failed  $(date +%H:%M:%S)"
du -sh "$ROOT"
