"""Bin the H8 residuals once, so the figure does not need the TauREx env.

matplotlib is not installed in `mirage-taurex` and FluxBinner is not installed in
`mirage`, so the binning and the plotting cannot happen in the same interpreter.
This runs in `mirage-taurex`, bins with the SAME FluxBinner the analysis uses, and
writes the residuals for scripts/fig_h8_three_level.py to read.

    PYTHONPATH=scripts conda run -n mirage-taurex python scripts/h8_residuals_for_fig.py
"""
import argparse
from pathlib import Path

import numpy as np
from h8_compare import binned, g395h_grid


def main(hifi, prt, published, out):
    a = np.load(hifi, allow_pickle=True)
    p = np.load(prt, allow_pickle=True)
    b = np.load(published, allow_pickle=True)
    wl, width = g395h_grid()
    tx_wl = 10000.0 / a["native_wn"]
    pub_wl = 10000.0 / b["native_wn"]

    code, lists = [], []
    for k in range(len(a["thetas"])):
        bt = binned(tx_wl, a["rprs2"][k], wl, width)
        code.append((binned(p["wl_um"], p["rprs2"][k], wl, width) - bt) * 1e6)
        lists.append((binned(pub_wl, b["rprs2"][k], wl, width) - bt) * 1e6)

    out = Path(out)
    np.savez_compressed(out, wl=wl, width=width,
                        code=np.asarray(code), line_list=np.asarray(lists))
    c, l = np.asarray(code), np.asarray(lists)
    print(f"[fig-data] {out}")
    print(f"  code      mean-draw offset {c[0].mean():+7.1f} ppm  shape rms {c[0].std():6.1f} ppm")
    print(f"  line list mean-draw offset {l[0].mean():+7.1f} ppm  shape rms {l[0].std():6.1f} ppm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hifi", default="data/h8/taurex_dump_k218.npz")
    ap.add_argument("--prt", default="data/h8/prt_spectra_k218.npz")
    ap.add_argument("--published", default="data/h8/taurex_dump_k218_published.npz")
    ap.add_argument("--out", default="data/h8/h8_residuals_k218.npz")
    a = ap.parse_args()
    main(a.hifi, a.prt, a.published, a.out)
