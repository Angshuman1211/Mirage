"""Fig 1, astronomy paper. The seven G395H reductions of one K2-18b visit,
overplotted on the shared grid, with the between-reduction spread below.

Unblocked by P6-D40. Every number is read from the canonical CSVs at plot time.
The grid is the one scripts/v5_h7_pipeline_shift.py:203 defines, geomspace(2.80,
5.10, 56), so 55 bins of which 53 are populated in every pair (P6-D36 corrects
the 60-bin figure that appeared in an earlier draft).
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANON = "data/v5_k218/canonical"
OUT = "figures/fig_v5h7_reductions.png"
G395H = ("G395H_NRS1", "G395H_NRS2", "NIRSPEC_G395H")
SCHMIDT = ("schmidt_exotedrf", "schmidt_eureka_ahrer", "schmidt_eureka_bell", "schmidt_firefly")
LUQUE = ("luque_exotedrf", "luque_eureka", "luque_sparta")


def grid():
    e = np.geomspace(2.80, 5.10, 56)
    return np.sqrt(e[:-1] * e[1:]), e


def read(name):
    wl, d, er = [], [], []
    with open(f"{CANON}/k218_combined_{name}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["instrument"] in G395H:
                wl.append(float(r["wavelength_um"]))
                d.append(float(r["transit_depth"]))
                er.append(float(r["depth_error"]))
    o = np.argsort(wl)
    return np.array(wl)[o], np.array(d)[o], np.array(er)[o]


def binned(wl, d, er, edges):
    idx = np.digitize(wl, edges) - 1
    n = len(edges) - 1
    out = np.full(n, np.nan)
    oute = np.full(n, np.nan)
    for b in range(n):
        m = idx == b
        if m.sum():
            w = 1.0 / er[m] ** 2
            out[b] = np.sum(d[m] * w) / np.sum(w)
            oute[b] = np.sqrt(1.0 / np.sum(w))
    return out, oute


def main():
    cen, edges = grid()
    names = SCHMIDT + LUQUE
    spec, errs = {}, {}
    for nm in names:
        wl, d, er = read(nm)
        spec[nm], errs[nm] = binned(wl, d, er, edges)

    M = np.vstack([spec[n] for n in names]) * 1e6          # ppm
    E = np.vstack([errs[n] for n in names]) * 1e6
    ok = np.isfinite(M).all(0)
    print(f"bins populated in all 7 reductions: {ok.sum()} of {len(cen)}")

    med = np.nanmedian(M, 0)
    resid = M - med
    # independent reproduction of the recorded spread, as a check on this binning
    pairspread = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dd = (M[i] - M[j])[ok]
            pairspread.append(np.sqrt(np.mean((dd - np.median(dd)) ** 2)))
    print(f"pairwise shape RMS over {len(pairspread)} pairs: "
          f"min {min(pairspread):.1f}  median {np.median(pairspread):.1f}  "
          f"max {max(pairspread):.1f} ppm")
    print(f"median single-bin error: {np.median(E[:, ok]):.1f} ppm")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.25]})
    cs = plt.cm.Blues(np.linspace(0.45, 0.9, len(SCHMIDT)))
    cl = plt.cm.Oranges(np.linspace(0.45, 0.9, len(LUQUE)))
    cols = {**dict(zip(SCHMIDT, cs)), **dict(zip(LUQUE, cl))}

    for nm in names:
        a1.plot(cen[ok], (spec[nm] * 1e6)[ok], lw=1.3, color=cols[nm],
                label=nm.replace("_", " "))
    a1.set_ylabel("transit depth (ppm)")
    a1.legend(ncol=2, fontsize=8.5, loc="upper left", framealpha=0.92)
    # "independent" alone overstates it. Seven reductions, two teams: Schmidt
    # contributes four pipelines and Luque three. The team count is the claim a
    # referee will test, so it belongs in the title (lesson from P6-D39).
    a1.set_title(f"{len(names)} reductions of one K2-18b G395H visit, two teams, shared grid",
                 fontsize=11)

    for k, nm in enumerate(names):
        a2.plot(cen[ok], resid[k][ok], lw=1.2, color=cols[nm])
    a2.axhline(0, color="0.4", lw=0.8, ls="--")
    eb = np.median(E[:, ok])
    a2.axhspan(-eb, eb, color="0.82", zorder=0)
    a2.annotate(f"median single-bin error  {eb:.0f} ppm", xy=(0.985, 0.06),
                xycoords="axes fraction", ha="right", fontsize=8.6,
                bbox=dict(fc="white", ec="0.7", alpha=0.92, pad=2.2))
    a2.set_ylabel("residual from\nper-bin median (ppm)")
    a2.set_xlabel("wavelength (um)")

    os.makedirs("figures", exist_ok=True)
    plt.subplots_adjust(left=0.098, right=0.985, top=0.935, bottom=0.095, hspace=0.09)
    plt.savefig(OUT, dpi=170)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
