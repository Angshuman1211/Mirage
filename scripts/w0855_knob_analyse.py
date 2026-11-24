"""What the background-subtraction knob moves, in the same units V5/H7 uses.

Offset is the constant part of the difference, shape is the RMS left after the
offset is removed. For an emission spectrum the analogue of a transit-depth ppm
is a fractional flux shift, so both are reported as ppm of the mean flux and
again in error-bar units.
"""
import numpy as np

d = np.load("data/jwst_w0855_cal/w0855_knob.npz")
wl, m_on, m_off, err = d["wl"], d["m_on"], d["m_off"], d["err_mean"]

good = np.isfinite(wl) & np.isfinite(m_on) & np.isfinite(m_off) & np.isfinite(err)
good &= (m_on > 0) & (err > 0)

# Miles figure 1: SNR per integration above 5 only beyond 3.6 um, and they quote
# mean SNR 73 over 3.827 to 5.170 um. Report their band and the full range.
bands = {"full (2.85-5.18 um)": good,
         "Miles SNR band (3.827-5.170 um)": good & (wl >= 3.827) & (wl <= 5.170)}

for name, sel in bands.items():
    f_on, f_off, e = m_on[sel], m_off[sel], err[sel]
    rel = (f_off - f_on) / f_on * 1e6          # ppm of flux
    off = np.median(rel)
    shape = np.sqrt(np.mean((rel - off) ** 2))
    # same thing in error-bar units
    rel_err = e / f_on * 1e6
    sig = np.median(rel_err)
    print(f"\n=== {name}   n={sel.sum()} bins ===")
    print(f"  median single-bin error          {sig:12.1f} ppm")
    print(f"  background OFFSET                {off:12.1f} ppm   "
          f"({abs(off)/sig:6.3f} sigma)")
    print(f"  background SHAPE RMS             {shape:12.1f} ppm   "
          f"({shape/sig:6.3f} sigma)")
    print(f"  shape as fraction of offset      {shape/abs(off)*100:11.1f} %")
    pos = (rel > 0).mean()
    print(f"  bins where background is positive{pos*100:11.1f} %")
    print(f"  mean SNR per bin                 {np.mean(f_on/e):12.1f}")

# how much flux is the background actually removing, absolutely
sel = bands["Miles SNR band (3.827-5.170 um)"]
print(f"\n=== absolute scale, Miles band ===")
print(f"  mean flux with background removed   {np.mean(m_on[sel]):.4e} MJy")
print(f"  mean flux without                   {np.mean(m_off[sel]):.4e} MJy")
print(f"  mean background removed             "
      f"{np.mean(m_off[sel]-m_on[sel]):.4e} MJy "
      f"({np.mean((m_off[sel]-m_on[sel])/m_on[sel])*100:.3f} % of source)")
