"""Builds the paper's result tables (CSV) from the run records.

usage: python3 make_tables.py [--runs results/runs] [--out results/tables]
"""
import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

SEEDS = (1234, 2345, 3456, 4567, 5678)
SEEDS3 = SEEDS[:3]
ROUNDS = tuple(range(1, 100, 2)) + (100,)
LAST = 10
WINDOW = {"y_from": ROUNDS[-LAST], "y_to": ROUNDS[-1]}
# archived names of five seed-1234 runs, used when the run is not found under its own name
ALIAS = {"q1_subfl_e16_s1234": "p_subfl_real", "q1_dense_e16_s1234": "p_dense", "q3_rand_res_s1234": "p_rand_res_real",
         "q3_rand_nores_s1234": "p_rand_nores_real", "rl_subfl_random_s1234": "p_subfl_random"}
# cells whose clips were selected at their own epsilon; the suffix names the clips
RETUNED = {("scratch", 32, "dense"): "_C8", ("scratch", 32, "subfl"): "_Ce8_Cr4", ("warm", 4, "dense"): "_C0.25",
           ("warm", 4, "subfl"): "_Ce0.5_Cr0.25", ("warm", 8, "dense"): "_C0.5", ("warm", 32, "subfl"): "_Ce2_Cr1"}
CONTROLS = (("q1_subfl_e16", "SubFL (anchor subspace + residual)", None),
            ("q3_subfl_nores", "SubFL without the residual", "q1_subfl_e16"),
            ("q3_rand_res", "random subspace + residual", "q1_subfl_e16"),
            ("q3_rand_nores", "random subspace, no residual", "q1_subfl_e16"),
            ("rl_subfl_random", "SubFL, random anchor labels", "q1_subfl_e16"),
            ("q1_dense_e16", "dense, C=4 (tuned)", "q1_subfl_e16"),
            ("q3_dense_c2", "dense, C=2 (= C_r)", "q1_subfl_e16"))
LEVERS = (("lev_const_subfl", "constant lr, SubFL", "q1_subfl_e16"),
          ("lev_const_dense", "constant lr, dense", "q1_dense_e16"),
          ("lev_r0.4_subfl", "eps split 0.4 / 0.6", "q1_subfl_e16"),
          ("lev_r0.6_subfl", "eps split 0.6 / 0.4", "q1_subfl_e16"))
NONIID = [(init, a) for init in ("scratch", "warm") for a in ("0.5", "0.1")]
PFL = ([("in", i, e, "") for i, e in (("scratch", 4), ("scratch", 16), ("warm", 4), ("warm", 16))]
       + [("in", i, 16, a) for i in ("scratch", "warm") for a in ("0.5", "0.1")]
       + [(v, "warm", e, "") for v in ("fn", "lp") for e in (4, 8, 16, 32)])
STUDIES = {"svhn": ("SVHN", "ResNet-20", (4, 16)), "r56": ("CIFAR-10", "ResNet-56", (16,))}
GIB = 2 ** 30


# paired t-test without scipy: regularized incomplete beta by continued fraction
def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for i in range(1, 300):
        m2 = 2 * i
        aa = i * (b - i) * x / ((qam + m2) * (a + m2))
        d = 1 / (1 + aa * d if abs(1 + aa * d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + i) * (qab + i) * x / ((a + m2) * (qap + m2))
        d = 1 / (1 + aa * d if abs(1 + aa * d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        if abs(d * c - 1) < 3e-14:
            break
    return h


def p_two_sided(t, df):
    x = df / (df + t * t)
    if x <= 0 or x >= 1:
        return float(x >= 1)
    a, b = df / 2, 0.5
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def t_quantile(q, df):
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_two_sided(mid, df) > 2 * (1 - q) else (lo, mid)
    return (lo + hi) / 2


def paired(a, b):
    d = [x - y for x, y in zip(a, b, strict=True)]
    n, md, sd = len(d), statistics.mean(d), statistics.stdev(d)
    half = t_quantile(0.975, n - 1) * sd / math.sqrt(n)
    p = p_two_sided(md / (sd / math.sqrt(n)), n - 1) if sd > 0 else float(md == 0)
    return {"n": n, "diff": md, "ci_lo": md - half, "ci_hi": md + half, "p": p, "wins": sum(x > 0 for x in d)}


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[i]["p"]))
        rows[i]["p_holm"] = running


def msd(ys):
    return {"mean": statistics.mean(ys), "sd": statistics.stdev(ys)}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def load(run_dir):
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(float(r["round"]) for r in rows) == ROUNDS, run_dir
    acc = [float(r["test_accuracy"]) for r in rows]
    g = [float(r["test_accuracy_global"]) for r in rows] if rows[0].get("test_accuracy_global") else None
    return {"dir": run_dir, "acc": acc, "y": sum(acc[-LAST:]) / LAST, "global": g,
            **{k: read_json(run_dir / f) for k, f in (("meta", "run_metadata.json"), ("man", "manifest.json"),
                                                      ("prov", "provenance.json"), ("status", "run_status.json"))}}


def hours(r):
    # start: provenance.json if present, else the run directory's timestamp (UTC)
    if "started" in r["prov"]:
        start = datetime.fromisoformat(r["prov"]["started"])
    else:
        start = datetime.strptime(r["dir"].name[:15], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    return (datetime.fromisoformat(r["status"]["updated"]) - start).total_seconds() / 3600


def clips(r, arm):
    keys = (("privacy", "clip_norm"),) if arm == "dense" else (("gep", "embedding_clip_norm"), ("gep", "residual_clip_norm"))
    return ",".join(f"{float(r['man'][s][k]):g}" for s, k in keys)


class Runs:
    """Run records under <root>/<study>/<run name>/<timestamp>/, by run name."""

    def __init__(self, root):
        found = {}
        for d in sorted(root.glob("*/*/*")):
            if (d / "metrics.csv").is_file():
                found.setdefault(d.parent.name, []).append(d)
        # of several attempts, the latest completed one
        self.index = {name: max(ds, key=lambda d: (read_json(d / "run_status.json").get("status") == "completed", d.name))
                      for name, ds in found.items()}
        self.used = {}

    def run(self, name):
        return load(self.index[name] if name in self.index else self.index[ALIAS[name]])

    def get(self, name):
        """Like run(), and lists the run in runs_used.csv."""
        if name not in self.used:
            self.used[name] = self.run(name)
        return self.used[name]

    def ys(self, group, seeds=SEEDS):
        return [self.get(f"{group}_s{s}")["y"] for s in seeds]


def main_study(runs):
    t = {}
    group = {(i, e, a): f"q{1 if i == 'scratch' else 2}_{a}_e{e}" + RETUNED.get((i, e, a), "")
             for i in ("scratch", "warm") for e in (4, 8, 16, 32) for a in ("dense", "subfl")}
    t["main_cells"] = [{"init": i, "eps": e, "arm": a, "group": g, "retuned": int((i, e, a) in RETUNED),
                        "clips": clips(runs.get(f"{g}_s1234"), a), "n": 5, **msd(runs.ys(g)),
                        **{f"y_s{s}": y for s, y in zip(SEEDS, runs.ys(g))}} for (i, e, a), g in sorted(group.items())]
    t["main_tests"] = [{"init": i, "eps": e, **paired(runs.ys(group[(i, e, "subfl")]),
                        runs.ys(group[(i, e, "dense")])), **WINDOW} for i in ("scratch", "warm") for e in (4, 8, 16, 32)]
    holm(t["main_tests"])
    t["ceilings"] = [{"init": i, "group": g, "n": 3, **msd(runs.ys(g, SEEDS3))}
                     for i, g in (("scratch", "c_nonprivate_lr0.3"), ("warm", "cw_nonprivate_lr0.03"))]
    t["noniid_tests"] = []
    for i, a in NONIID:
        s, d = runs.ys(f"q5_{i}_a{a}_subfl"), runs.ys(f"q5_{i}_a{a}_dense")
        t["noniid_tests"].append({"init": i, "alpha": a, "subfl_mean": statistics.mean(s),
                                  "subfl_sd": statistics.stdev(s), "dense_mean": statistics.mean(d),
                                  "dense_sd": statistics.stdev(d), **paired(s, d), **WINDOW})
    holm(t["noniid_tests"])
    for key, spec in (("controls", CONTROLS), ("levers", LEVERS)):
        t[key] = []
        for name, label, ref in spec:
            row = {"name": name, "label": label, "n": 3, "seeds": "1234 2345 3456", **msd(runs.ys(name, SEEDS3)),
                   "reference": ref or ""}
            if ref:
                p = paired(runs.ys(name, SEEDS3), runs.ys(ref, SEEDS3))
                row.update({k: p[k] for k in ("diff", "ci_lo", "ci_hi", "p")})
            t[key].append(row)
    gap = lambda s, d: [x - y for x, y in zip(runs.ys(s, SEEDS3), runs.ys(d, SEEDS3))]  # noqa: E731
    did = paired(gap("lev_const_subfl", "lev_const_dense"), gap("q1_subfl_e16", "q1_dense_e16"))
    t["levers"].append({"name": "gap_did", "label": "gap under constant lr - gap under cosine lr", "n": 3,
                        "seeds": "1234 2345 3456", "mean": "", "sd": "", "reference": "q1_subfl_e16 - q1_dense_e16",
                        **{k: did[k] for k in ("diff", "ci_lo", "ci_hi", "p")}})
    settings = [(f"iid_{i}_e16", group[(i, 16, a)], a) for i in ("scratch", "warm") for a in ("subfl", "dense")]
    settings += [(f"noniid_{i}_a{al}", f"q5_{i}_a{al}_{a}", a) for i, al in NONIID for a in ("subfl", "dense")]
    t["curves"] = [{"setting": st, "arm": a, "seed": s, "round": rnd, "acc": acc} for st, g, a in settings
                   for s in SEEDS for rnd, acc in zip(ROUNDS, runs.get(f"{g}_s{s}")["acc"])]
    t["runs_used"] = [{"name": n, "run": r["dir"].parent.name, "dir": "/".join(r["dir"].parts[-3:]), "y": r["y"],
                       "eps": r["meta"].get("record_level_epsilon", r["meta"].get("achieved_epsilon")),
                       "target": r["man"]["privacy"]["total_epsilon"], "dp_valid": r["meta"].get("dp_valid")}
                      for n, r in sorted(runs.used.items())]
    # wall time per run: run_status.json "updated" minus the start (shipped records: the run directory's timestamp)
    h = {a: [hours(runs.get(f"{g}_s{s}")) for (_, _, arm), g in group.items() if arm == a for s in SEEDS]
         for a in ("subfl", "dense")}
    t["runtime"] = [{"arm": a, "n": len(h[a]), "run_h_mean": statistics.mean(h[a]),
                     "ratio": statistics.mean(h[a]) / statistics.mean(h["dense"])} for a in ("subfl", "dense")]
    return t, group


def pfl(runs, group):
    cells, tests = [], []
    for v, i, e, a in PFL:
        rs = [runs.run(f"pf{e}_{v}_{i}{'_a' + a if a else ''}_s{s}") for s in SEEDS]
        # ys: mean over clients of their deployed models; ys_global: the global model alone
        ys = [statistics.mean(r["acc"][-LAST:]) for r in rs]
        ys_global = [statistics.mean(r["global"][-LAST:]) for r in rs] if v != "lp" else None
        ours = {arm: runs.ys(group[(i, e, arm)] if not a else f"q5_{i}_a{a}_{arm}") for arm in ("subfl", "dense")}
        key = {"variant": v, "init": i, "eps": e, "alpha": a, "setting": f"dirichlet_a{a}" if a else "iid"}
        cells.append({**key, "clip": f"{rs[0]['man']['privacy']['clip_norm']:g}", "n": 5,
                      "subfl_mean": statistics.mean(ours["subfl"]), "subfl_sd": statistics.stdev(ours["subfl"]),
                      "dense_mean": statistics.mean(ours["dense"]), "dense_sd": statistics.stdev(ours["dense"]),
                      "pfl_mean": statistics.mean(ys), "pfl_sd": statistics.stdev(ys),
                      "pfl_global_mean": statistics.mean(ys_global) if ys_global else "",
                      "pfl_global_sd": statistics.stdev(ys_global) if ys_global else "",
                      **{f"pfl_s{s}": y for s, y in zip(SEEDS, ys)}})
        p, pd = paired(ours["subfl"], ys), paired(ys, ours["dense"])
        tests.append({**key, **p, "m": 8, "p_holm": None, **WINDOW, "pfl_dense_diff": pd["diff"],
                      "pfl_dense_ci_lo": pd["ci_lo"], "pfl_dense_ci_hi": pd["ci_hi"],
                      "pfl_minus_global_diff": statistics.mean(ys) - statistics.mean(ys_global) if ys_global else ""})
    holm(tests[:8])  # the 8 whole-model ("in") cells
    holm(tests[8:])  # the 8 frozen-backbone ("fn", "lp") cells
    return {"pfl_cells": cells, "pfl_tests": tests}


def study(runs, name):
    dataset, model, epss = STUDIES[name]
    label = {"study": name, "dataset": dataset, "model": model}
    used = {(e, a, s): runs.run(f"{name}_e{e}_{a}_s{s}") for e in epss for a in ("subfl", "dense") for s in SEEDS}
    cells, tests, cost = [], [], []
    for e in epss:
        for a in ("dense", "subfl"):
            ys = [used[(e, a, s)]["y"] for s in SEEDS]
            cells.append({**label, "eps": e, "arm": a, "clips": clips(used[(e, a, 1234)], a), "n": 5, **msd(ys),
                          **{f"y_s{s}": y for s, y in zip(SEEDS, ys)}})
        p = paired(*([used[(e, a, s)]["y"] for s in SEEDS] for a in ("subfl", "dense")))
        tests.append({**label, "eps": e, **p, **WINDOW, "m": len(epss), "exploratory": 1})
    holm(tests)
    steps = sum(c["steps_per_round"] for c in used[(epss[0], "subfl", 1234)]["meta"]["clients"])
    for a in ("subfl", "dense"):
        for pooled in [(e,) for e in epss] + ([epss] if len(epss) > 1 else []):
            cs = [r["meta"]["cost"] for (e, arm, _), r in used.items() if arm == a and e in pooled]
            alloc = [c["gpu_peak_memory_allocated_bytes"] / GIB for c in cs]
            res = [c["gpu_peak_memory_reserved_bytes"] / GIB for c in cs]
            rnd = [statistics.mean(c["round_seconds"]) for c in cs]
            hrs = [c["wall_seconds"] / 3600 for c in cs]
            cost.append({**label, "arm": a, "eps": " ".join(map(str, pooled)), "n": len(cs),
                         "peak_allocated_gib_mean": statistics.mean(alloc), "peak_allocated_gib_max": max(alloc),
                         "peak_reserved_gib_mean": statistics.mean(res), "peak_reserved_gib_max": max(res),
                         "round_s_mean": statistics.mean(rnd), "round_s_sd": statistics.stdev(rnd), "round_steps": steps,
                         "step_s_mean": statistics.mean(rnd) / steps,
                         "evaluation_s_mean": statistics.mean(statistics.mean(c["evaluation_seconds"]) for c in cs),
                         "run_h_mean": statistics.mean(hrs), "run_h_max": max(hrs)})
    return {f"{name}_cells": cells, f"{name}_tests": tests, f"{name}_cost": cost}


def write(tables, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: format(v, ".3g" if k in ("p", "p_holm") else ".6f") if isinstance(v, float) else v
                            for k, v in r.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("results/runs"), help="run records: <study>/<run name>/<timestamp>/")
    ap.add_argument("--out", type=Path, default=Path("results/tables"), help="where the CSVs are written")
    args = ap.parse_args()
    runs = Runs(args.runs)
    tables, group = main_study(runs)
    tables.update(pfl(runs, group))
    for s in STUDIES:
        tables.update(study(runs, s))
    write(tables, args.out)
    print(f"wrote {len(tables)} tables to {args.out}")
