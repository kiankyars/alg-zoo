"""Markdown tables for README.md from results/ (n = 3 sweep, exact table, sliver table)."""

import glob
import json
import os

import numpy as np

from summarize import summary


def sweep_table():
    rows = []
    for f in glob.glob("results/test/M_*_3_*_lguard.jsonl"):
        d = int(f.split("/")[-1].split("_")[1])
        sigma = f.split("/")[-1].split("_")[3]
        s = summary(f)
        rows.append((d, float(sigma), s))
    rows.sort(key=lambda r: (r[0], r[1]))
    out = ["| model | params | σ | noisy models | mean accuracy | exact cost (fwd) | worst fair ratio | 90% CI | bootstrap share > 1 | binding rung (at budget) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for d, sig, s in rows:
        p = d + d * d + 3 * d
        mark = "**" if s["fair_hi"] < 1 else ""
        rung, at = s["binding"] if s["binding"] else ("-", 0)
        out.append(f"| `M_{{{d},3}}` | {p} | {sig} | {s['K']} | {s['mean_acc']:.4f} | {s['exact_cost']:.0f} | "
                   f"{mark}{s['fair']:.2f}{mark} | [{s['fair_lo']:.2f}, {s['fair_hi']:.2f}] | {s['p_over']:.2f} | {rung} ({at:.0f}) |")
    return "\n".join(out)


def dev_table():
    cells = {}
    for f in glob.glob("results/dev/M_*_3_*_*.jsonl"):
        parts = f.split("/")[-1][:-6].split("_")
        d, sigma, rule = int(parts[1]), parts[3], parts[4]
        cells[(d, sigma, rule)] = summary(f, boot=400)["fair"]
    rules = ["tie", "cguard", "lguard"]
    out = ["| model | σ | " + " | ".join(rules) + " |", "|---|---|" + "---|" * len(rules)]
    for d in sorted({k[0] for k in cells}, reverse=True):
        for sigma in sorted({k[1] for k in cells}, key=float):
            vals = [cells.get((d, sigma, ru)) for ru in rules]
            out.append(f"| `M_{{{d},3}}` | {sigma} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in vals) + " |")
    return "\n".join(out)


def _rows(pattern):
    """Rows of all matching result files, one per model (from the most recently written file)."""
    best = {}
    for p in sorted(glob.glob(pattern), key=os.path.getmtime):
        for r in json.load(open(p)):
            best[(r["family"], r["d"], r["n"])] = r
    return list(best.values())


def exact_table():
    rows = sorted(_rows("results/exact_table_n*.json"), key=lambda r: (r["family"] != "M", r["n"], r["d"]))
    if not rows:
        return "(pending)"
    out = ["| model | params | exact accuracy | change at higher order | cost (fwd) | Monte Carlo (2^24) | z |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        name = ("M" if r["family"] == "M" else "AM") + f"_{{{r['d']},{r['n']}}}"
        out.append(f"| `{name}` | {r['params']} | {r['exact']:.10f} | "
                   f"{'not run' if r['measure_disagreement'] is None else format(r['measure_disagreement'], '.0e')} | "
                   f"{r['cost_fwd']:.3g} | {r['mc']:.6f} | {r['z']:+.2f} |")
    return "\n".join(out)


def _rel(x):
    return f"{x:+.1e}" if abs(x) < 1e-4 else f"{x:+.2%}"


def sliver_table():
    rows = sorted(_rows("results/sliver_table_n*.json"), key=lambda r: (r["n"], r["family"] != "M", r["d"]))
    if not rows:
        return "(pending)"
    decomp = {}
    if os.path.exists("results/sliver_decomposition.json"):
        decomp = {(r["family"], r["d"], r["n"]): r for r in json.load(open("results/sliver_decomposition.json"))}
    out = ["| model | exact error | first-order law | relative difference | order-truncated law | "
           "relative difference | wrong inputs in one / two / no slivers | gap bending |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = ("M" if r["family"] == "M" else "AM") + f"_{{{r['d']},{r['n']}}}"
        dc = decomp.get((r["family"], r["d"], r["n"]))
        shares = bend = "-"
        if dc is not None:
            shares = " / ".join(f"{dc[k] / dc['error']:.1%}" for k in ("one", "two", "none"))
            # truncated law - exact = (law - sum of true slivers) + overlaps - errors in no sliver
            bend = f"{r['truncated_rel_diff'] - (dc['two'] - dc['none']) / dc['error']:+.1%}"
        out.append(f"| `{name}` | {r['exact_error']:.4e} | {r['sliver_error']:.4e} | {_rel(r['sliver_rel_diff'])} | "
                   f"{r['truncated_error']:.4e} | {_rel(r['truncated_rel_diff'])} | {shares} | {bend} |")
    return "\n".join(out)


def seeds_table():
    if not os.path.exists("results/sliver_seeds.json"):
        return "(pending)"
    rows = json.load(open("results/sliver_seeds.json"))
    out = ["| model | seed | error | how | first-order law | order-truncated law | "
           "wrong inputs in one / two / no slivers |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["n"], r["family"] != "M", r["d"], r["seed"])):
        name = ("M" if r["family"] == "M" else "AM") + f"_{{{r['d']},{r['n']}}}"
        shares = " / ".join(f"{r[k]:.1%}" for k in ("one", "two", "none")) if "two" in r else "-"
        how = "exact" if r["method"] == "exact" else f"MC ±{r['error_se'] / r['error']:.1%}"
        out.append(f"| `{name}` | {r['seed']} | {r['error']:.3e} | {how} | "
                   f"{_rel((r['first_order'] - r['error']) / r['error'])} | "
                   f"{_rel((r['truncated'] - r['error']) / r['error'])} | {shares} |")
    return "\n".join(out)


def fill_readme(path="README.md"):
    """Replace the content between <!-- KEY --> and <!-- /KEY --> markers with fresh tables."""
    import re

    text = open(path).read()
    for key, fn in [("SWEEP_TABLE", sweep_table), ("DEV_TABLE", dev_table), ("EXACT_TABLE", exact_table),
                    ("SLIVER_TABLE", sliver_table), ("SEEDS_TABLE", seeds_table)]:
        text = re.sub(rf"<!-- {key} -->.*?<!-- /{key} -->", lambda _: f"<!-- {key} -->\n{fn()}\n<!-- /{key} -->",
                      text, flags=re.S)
    open(path, "w").write(text)


if __name__ == "__main__":
    import sys

    if "--readme" in sys.argv:
        fill_readme()
    else:
        print("## sweep\n\n" + sweep_table() + "\n\n## exact\n\n" + exact_table() + "\n\n## sliver\n\n" + sliver_table()
              + "\n\n## seeds\n\n" + seeds_table())
