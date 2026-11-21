"""V7 summary: does the translation null survive a real side built from all three
reduced spectra?

Reads configs/cyclegan_v7_pooled3/ablation_{target}_{full,n56}.json (written by
run_v7_pooled.sh) and the published per-target runs, and prints one line per
target plus the decision-log sentence.

Like-for-like only. P5-D11's control showed the CycleGAN's MMD falls 71% from
scoring against fewer real frames, so the pooled translator at 56 frames is set
beside the published tier-2 translator at 56 frames, and the full-frame scoring
beside the full-frame control (WASP-39b only has one).

The verdict rests on the FULL-frame rows, with both metrics required to agree:
  holds     translation worse than randomisation on MMD and KL
  weakens   worse than randomisation, but now better than no adaptation
  reverses  better than randomisation on both

    python scripts/v7_table.py
"""
import json
from pathlib import Path

RUN = Path("configs/cyclegan_v7_pooled3")
PUBLISHED = {("wasp39", "n56"): "cyclegan_t2_wasp39",
             ("wasp96", "n56"): "cyclegan_t2_wasp96",
             ("k218", "n56"): "cyclegan_t2_k218",
             ("wasp39", "full"): "cyclegan_ctrl_wasp39_26b_full"}
LABEL = {"wasp39": "WASP-39b PRISM", "wasp96": "WASP-96b NIRISS", "k218": "K2-18b NIRISS"}


def load(p: Path):
    return json.load(open(p))["conditions"] if p.exists() else None


def load_published(run: str):
    """The published per-target numbers, on the SAME MMD basis as the pooled run.

    `ablation.json` was scored with the basis fitted on all four arms, including
    the translated output, so its MMD is checkpoint-dependent and is not
    comparable to anything scored on another checkpoint. `ablation_fixedbasis.json`
    is the same checkpoint re-scored with the basis fitted on sim and real only.
    Prefer it, and say so if it is absent rather than silently mixing bases.
    """
    d = Path("configs") / run
    fx = d / "ablation_fixedbasis.json"
    if fx.exists():
        return load(fx), True
    return load(d / "ablation.json"), False


def verdict(c):
    s, r, g = c["sim_baseline"], c["domain_random"], c["cyclegan_trans"]
    worse_r = g["mmd_to_real"] > r["mmd_to_real"] and g["kl_to_real"] > r["kl_to_real"]
    better_r = g["mmd_to_real"] < r["mmd_to_real"] and g["kl_to_real"] < r["kl_to_real"]
    better_s = g["mmd_to_real"] < s["mmd_to_real"] and g["kl_to_real"] < s["kl_to_real"]
    if better_r:
        return "reverses"
    if worse_r:
        return "weakens" if better_s else "holds"
    return "split"


def main():
    print(f"\n{'=' * 100}\n  V7: one translator trained on all three real spectra, "
          f"26 shared bins\n{'=' * 100}")
    print(f"  {'target':<17}{'frames':<7}{'MMD sim':>9}{'random':>9}{'pooled GAN':>12}"
          f"{'published GAN':>15}{'KL random':>11}{'pooled':>9}{'shift':>8}  verdict")
    print("  " + "-" * 96)
    verdicts = {}
    stale = False
    for t in LABEL:
        for n in ("full", "n56"):
            c = load(RUN / f"ablation_{t}_{n}.json")
            if c is None:
                print(f"  {LABEL[t]:<17}{n:<7}  [missing] run scripts/run_v7_pooled.sh")
                continue
            pub = PUBLISHED.get((t, n))
            pc, matched = load_published(pub) if pub else (None, True)
            if pc is None:
                pg = "--"
            else:
                pg = f"{pc['cyclegan_trans']['mmd_to_real']:.4f}"
                if not matched:
                    pg += "*"
                    stale = True
            v = verdict(c)
            if n == "full":
                verdicts[t] = v
            print(f"  {LABEL[t]:<17}{n:<7}{c['sim_baseline']['mmd_to_real']:>9.4f}"
                  f"{c['domain_random']['mmd_to_real']:>9.4f}"
                  f"{c['cyclegan_trans']['mmd_to_real']:>12.4f}{pg:>15}"
                  f"{c['domain_random']['kl_to_real']:>11.3f}"
                  f"{c['cyclegan_trans']['kl_to_real']:>9.3f}"
                  f"{c['cyclegan_trans']['mean_shift']:>8.3f}  {v}")
    print("  " + "-" * 96)
    if stale:
        print("  * published number is on the OLD checkpoint-dependent MMD basis and is "
              "not comparable.\n    Re-score it with run_cyclegan_ablation_standalone.py "
              "to produce ablation_fixedbasis.json.")
    if verdicts:
        vs = set(verdicts.values())
        overall = vs.pop() if len(vs) == 1 else "mixed (" + ", ".join(
            f"{LABEL[t]} {v}" for t, v in verdicts.items()) + ")"
        print(f"\n  decision-log sentence: with the real side rebuilt from all three "
              f"reduced spectra, the translation null {overall} on full-frame scoring.")


if __name__ == "__main__":
    main()
