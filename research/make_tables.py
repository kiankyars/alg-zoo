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
    """The sliver laws, and the exact decomposition of the order-truncated law's error:
    truncated - exact = overlaps (truncated - bisector) + bending (bisector - exact sliver law)
    - unattached (exact - exact sliver law), all as fractions of the exact error; then the Monte
    Carlo shares of the wrong inputs in one / two / no slivers (sliver_mc.py table)."""
    rows = sorted(_rows("results/sliver_table_n*.json"), key=lambda r: (r["n"], r["family"] != "M", r["d"]))
    if not rows:
        return "(pending)"
    mc, decomp = {}, {}
    if os.path.exists("results/sliver_widths.json"):  # Monte Carlo exact sliver law where not computed
        mc = {(r["family"], r["d"], r["n"]): r for r in json.load(open("results/sliver_widths.json"))}
    if os.path.exists("results/sliver_decomposition.json"):  # Monte Carlo: wrong inputs in 1 / 2 / 0 slivers
        decomp = {(r["family"], r["d"], r["n"]): r for r in json.load(open("results/sliver_decomposition.json"))}
    out = ["| model | exact error | first-order law | order-truncated law | bisector law | exact sliver law | "
           "overlaps | bending | unattached | in one / two / no slivers |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        key = (r["family"], r["d"], r["n"])
        name = ("M" if r["family"] == "M" else "AM") + f"_{{{r['d']},{r['n']}}}"
        ex = r["exact_error"]
        cells = [f"`{name}`", f"{ex:.4e}"] + [_rel(r[f"{k}_rel_diff"]) if f"{k}_rel_diff" in r else "-"
                                              for k in ("sliver", "truncated", "bisector")]
        es, note = r.get("exact_sliver_error"), ""
        if es is None and key in mc:
            es, note = mc[key]["exact_sliver"], f" (MC ±{mc[key]['exact_sliver_se'] / ex:.1%})"
        if es is None:
            cells += ["-"] * 4
        else:
            cells += [_rel((es - ex) / ex) + note, _rel((r["truncated_error"] - r["bisector_error"]) / ex),
                      _rel((r["bisector_error"] - es) / ex), _rel((ex - es) / ex)]
        dc = decomp.get(key)
        cells.append(" / ".join(f"{dc[k] / dc['error']:.1%}" for k in ("one", "two", "none")) if dc else "-")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def seeds_table():
    if not os.path.exists("results/sliver_seeds.json"):
        return "(pending)"
    rows = json.load(open("results/sliver_seeds.json"))
    out = ["| model | seed | error | how | first-order law | order-truncated law | bisector law | "
           "exact sliver law | in one / two / no slivers |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["n"], r["family"] != "M", r["d"], r["seed"])):
        name = ("M" if r["family"] == "M" else "AM") + f"_{{{r['d']},{r['n']}}}"
        how = "exact" if r["method"] == "exact" else f"MC ±{r['error_se'] / r['error']:.1%}"
        rel = lambda k: _rel((r[k] - r["error"]) / r["error"]) if k in r else "-"
        shares = " / ".join(f"{r[k]:.1%}" for k in ("one", "two", "none")) if "two" in r else "-"
        out.append(f"| `{name}` | {r['seed']} | {r['error']:.3e} | {how} | {rel('first_order')} | "
                   f"{rel('truncated')} | {rel('bisector')} | {rel('exact_sliver')} | {shares} |")
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
