#!/usr/bin/env python3
"""Recompute every result table from the generation rows in results/rows/.

No GPU, no model: only pandas and the shipped l2.py (for the MGSM answer parser).

    python scripts/readout.py                 # prints all tables, writes results/tables/*.csv
    python scripts/readout.py --tag trim_a100 # one comparison only

Row files: results/rows/<model>/<tag>/<dataset>__<variant>__<precision>.jsonl
  model = l2thinker_trim (trimmed) | l2thinker_ctrl (untrimmed control, same session)
  tags  = trim_a100 (round 1), trim_a100_rerun (E0, control second pass),
          trim_a100_r2 (E1/E2), trim_a100_r2x (exploratory MGSM sw)
"""
import argparse, glob, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
import l2  # noqa: E402  (vendor/l2.py: parser + metrics, stdlib only)

ROWS = ROOT / "results" / "rows"
TABLES = ROOT / "results" / "tables"
KEY = ["dataset", "precision", "lang", "id"]
TRIM, CTRL = "l2thinker_trim", "l2thinker_ctrl"

COMPARISONS = {  # tag -> (label, in-region languages)
    "trim_a100": ("Round 1: Bengali MGSM, Bengali/English Aya, f16 + q4_k_m", {"bn", "en"}),
    "trim_a100_r2": ("Round 2 E1/E2: Telugu MGSM, Hindi/Gujarati/Tamil/Swahili Aya, f16", {"hi", "gu", "ta", "te"}),
    "trim_a100_r2x": ("Exploratory: Swahili MGSM on the South Asia trim, f16", set()),
}


def rows_df(model, tag):
    files = sorted(glob.glob(str(ROWS / model / tag / "*.jsonl")))
    recs = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(recs)
    if len(df):
        m = df.dataset == "mgsm"  # rescore with the shipped parser so results don't depend on the generating session
        df.loc[m, "correct"] = [l2.mgsm_correct(a, g) for a, g in zip(df.loc[m, "answer"], df.loc[m, "gold"])]
    return df


def mcnemar_p(a, b):
    """Exact two-sided McNemar on discordant pairs (binomial)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum()); n = n01 + n10
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n)


def bootstrap_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pair(tag):
    ct, tr = rows_df(CTRL, tag), rows_df(TRIM, tag)
    assert len(ct) and len(tr), f"no rows for tag {tag}"
    assert not ct.duplicated(KEY).any() and not tr.duplicated(KEY).any(), "duplicate rows"
    gpus = set(ct.gpu) | set(tr.gpu); assert len(gpus) == 1, f"mixed GPUs {gpus}: paired comparisons need one GPU type"
    assert set(ct.decoding) | set(tr.decoding) == {"greedy"}, "identity is only meaningful under greedy decoding"
    p = ct.merge(tr, on=KEY, suffixes=("_c", "_t"), validate="1:1")
    assert len(p) == len(ct) == len(tr), (len(p), len(ct), len(tr))
    p["same"] = (p.trace_c == p.trace_t) & (p.answer_c == p.answer_t)
    return p, gpus.pop()


def comparison_table(p, in_region):
    out = []
    for (ds, prec, lang), g in p.groupby(["dataset", "precision", "lang"]):
        r = dict(dataset=ds, precision=prec, lang=lang, in_region=lang in in_region, n=len(g),
                 identity_pct=100 * g.same.mean(),
                 trunc_ctrl_pct=100 * g.truncated_c.mean(), d_trunc_pt=100 * (g.truncated_t.mean() - g.truncated_c.mean()),
                 loop_ctrl=g.doomloop_c.mean(), d_loop=g.doomloop_t.mean() - g.doomloop_c.mean())
        if "l2_c" in g:
            r.update(l2_ctrl_pct=100 * g.l2_c.mean(), d_l2_pt=100 * (g.l2_t.mean() - g.l2_c.mean()))
        if ds == "mgsm":
            a, b = g.correct_c.astype(bool).values, g.correct_t.astype(bool).values
            r.update(acc_ctrl_pct=100 * a.mean(), acc_trim_pct=100 * b.mean(), d_acc_pt=100 * (b.mean() - a.mean()),
                     discordant=int((a != b).sum()), mcnemar_p=mcnemar_p(a, b))
        if len(g) >= 50:
            lo, hi = bootstrap_ci(g.doomloop_t - g.doomloop_c); r.update(d_loop_ci_lo=lo, d_loop_ci_hi=hi)
        out.append(r)
    return pd.DataFrame(out)


def checks(res, in_region):
    """Pre-registered bounds, applied to in-region groups only."""
    r = res[res.in_region] if in_region else res
    lines = []
    if len(r):
        f16 = r[r.precision == "f16"]
        lines.append(("f16 identity >= 98%", f16.identity_pct.round(2).tolist(), bool((f16.identity_pct >= 98).all())))
        for col, tol in (("d_trunc_pt", 1), ("d_acc_pt", 1), ("d_l2_pt", 1), ("d_loop", 0.01)):
            if col in r:
                v = r[col].dropna(); lines.append((f"{col} within ±{tol}", v.round(3).tolist(), bool((v.abs() <= tol).all())))
        if "mcnemar_p" in r:
            v = r.mcnemar_p.dropna(); lines.append(("McNemar n.s. (p > 0.05)", v.round(3).tolist(), bool((v > 0.05).all())))
    return lines


def noise_floor():
    c1, c2 = rows_df(CTRL, "trim_a100"), rows_df(CTRL, "trim_a100_rerun")
    q = c1.merge(c2, on=KEY, suffixes=("_1", "_2"), validate="1:1")
    q["same"] = (q.trace_1 == q.trace_2) & (q.answer_1 == q.answer_2)
    t = q.groupby(["dataset", "precision", "lang"]).same.mean().mul(100).rename("ctrl_vs_ctrl_identity_pct").reset_index()
    return t, 100 * q.same.mean(), len(q)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=None); a = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    tags = [a.tag] if a.tag else list(COMPARISONS)
    for tag in tags:
        label, in_region = COMPARISONS.get(tag, (tag, set()))
        p, gpu = pair(tag)
        res = comparison_table(p, in_region)
        print(f"\n== {label}\n   tag={tag} gpu={gpu} paired rows={len(p)} identical={int(p.same.sum())} ({100 * p.same.mean():.2f}%)")
        print(res.round(3).to_string(index=False))
        for name, vals, ok in checks(res, bool(in_region)):
            print(f"   {name}: {vals} -> {'PASS' if ok else 'FAIL'}")
        res.to_csv(TABLES / f"{tag}.csv", index=False)
        p.loc[~p.same, KEY].to_csv(TABLES / f"{tag}_divergent_ids.csv", index=False)
    if not a.tag or a.tag == "trim_a100":
        t, ident, n = noise_floor()
        print(f"\n== E0 noise floor: control vs control (second pass, different A100 instance)\n{t.round(1).to_string(index=False)}")
        print(f"   overall {ident:.2f}% identical over {n} rows")
        t.to_csv(TABLES / "e0_noise_floor.csv", index=False)
    print(f"\ntables written to {TABLES}")


if __name__ == "__main__":
    main()
