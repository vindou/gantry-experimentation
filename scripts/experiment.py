"""Run a grid-search or ablation experiment defined in a YAML file.

    python scripts/experiment.py run experiments/ablation.yaml --gpus 0,1,2,3
    python scripts/experiment.py summarize experiments/ablation.yaml
    python scripts/experiment.py run experiments/scale_grid_stage1.yaml --dry-run

Experiment YAML
---------------
    name: my_experiment                  # results in runs/<name>/<variant>/seed<k>/
    env_config: configs/gantry.yaml
    ppo_config: configs/ppo.yaml
    total_timesteps: 20_000_000          # optional, overrides the PPO config
    seeds: [0, 1, 2]
    base: {reward.clf.w_vdot: 0.0}       # dotted env-config overrides for every variant
    grid:                                # cartesian product -> variants "sigma_v=0.1"
      reward.clf.sigma_v: [0.05, 0.1, null]
    variants:                            # and/or named variants
      control: {reward.clf.w_vdot: 0.0}
      pd: {_policy: pd}                  # classical baseline, no training
    eval:
      episodes: 128
      conditions:                        # each condition = one evaluate.py call
        in_dist: {}
        nominal: {no_randomization: true}
        payload: {shift: {m_x: 1.5}}
        hifi: {integrator: kvaerno5, x64: true, episodes: 32}
    objective:                           # used to rank variants in the summary
      condition: in_dist
      metric: rms_err_mm_mean
      termination_penalty_mm: 10.0       # score += penalty * termination_rate

Each (variant, seed) is one job: train, then evaluate every condition. Jobs
that already have results are skipped, so an interrupted experiment resumes.
With --gpus, jobs run in parallel, one GPU each (CUDA_VISIBLE_DEVICES).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import itertools
import json
import os
import queue
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODES = ("setpoint", "trajectory")


# ------------------------------------------------------------------ experiment spec


def _fmt(v) -> str:
    return "formula" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def load_experiment(path: Path, cli_sets: dict) -> dict:
    exp = yaml.safe_load(path.read_text())
    exp.setdefault("base", {})
    exp["base"] = {**(exp["base"] or {}), **cli_sets}
    variants: dict[str, dict] = {}
    grid = exp.get("grid") or {}
    if grid:
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            name = ",".join(f"{k.split('.')[-1]}={_fmt(v)}" for k, v in zip(keys, values))
            variants[name] = dict(zip(keys, values))
    for name, ov in (exp.get("variants") or {}).items():
        variants[name] = dict(ov or {})
    if not variants:
        raise ValueError("experiment needs `grid` and/or `variants`")
    exp["_variants"] = variants
    exp.setdefault("eval", {})
    exp["eval"].setdefault("episodes", 128)
    exp["eval"].setdefault("conditions", {"in_dist": {}})
    exp.setdefault("objective", {"condition": "in_dist", "metric": "rms_err_mm_mean",
                                 "termination_penalty_mm": 10.0})
    return exp


def jobs_of(exp: dict, out: Path):
    for vname, ov in exp["_variants"].items():
        is_pd = ov.get("_policy") == "pd"
        seeds = [exp["seeds"][0]] if is_pd else exp["seeds"]  # PD is deterministic
        for seed in seeds:
            yield dict(variant=vname, seed=seed, overrides={**exp["base"], **ov},
                       run_dir=out / vname / f"seed{seed}", pd=is_pd)


def check_resolved(exp: dict, out: Path, force: bool) -> None:
    """Refuse to mix results from a different experiment definition under one name."""
    resolved = {k: v for k, v in exp.items() if not k.startswith("_")}
    resolved["variants_resolved"] = exp["_variants"]
    text = yaml.safe_dump(resolved, sort_keys=True)
    f = out / "experiment.yaml"
    if f.exists() and f.read_text() != text and not force:
        sys.exit(f"{f} exists with a different definition; use --name or --force")
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(text)


# ----------------------------------------------------------------------- running


def _sets_args(overrides: dict) -> list[str]:
    args = []
    for k, v in overrides.items():
        if not k.startswith("_"):
            args += ["--set", f"{k}={json.dumps(v)}"]
    return args


def run_job(job: dict, exp: dict, gpu: str | None, args) -> str:
    run_dir: Path = job["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # several jobs may share a device
    py = sys.executable
    t0 = time.time()

    # --- train (or, for PD, just write the env config the evaluator reads)
    if job["pd"]:
        if not (run_dir / "gantry.yaml").exists():
            sys.path.insert(0, str(ROOT))
            from gantry_sim.params import apply_overrides, load_yaml
            d = apply_overrides(load_yaml(ROOT / exp["env_config"]),
                                {k: v for k, v in job["overrides"].items() if not k.startswith("_")})
            (run_dir / "gantry.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    elif not (run_dir / "models.eqx").exists():
        cmd = [py, "scripts/train.py", "--env-config", exp["env_config"],
               "--ppo-config", args.ppo_config or exp["ppo_config"],
               "--run-dir", str(run_dir), "--seed", str(job["seed"]), *_sets_args(job["overrides"])]
        steps = args.total_timesteps or exp.get("total_timesteps")
        if steps:
            cmd += ["--total-timesteps", str(steps)]
        with open(run_dir / "train.log", "w") as log:
            if subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode:
                return f"FAILED training {run_dir} (see train.log)"

    # --- evaluate: conditions that share integrator / precision / episode count
    # run in ONE evaluator process with a single compilation
    groups: dict[tuple, list] = {}
    for cname, cond in exp["eval"]["conditions"].items():
        if (run_dir / "eval" / f"{cname}.json").exists():
            continue
        cond = cond or {}
        episodes = args.episodes or cond.get("episodes", exp["eval"]["episodes"])
        key = (cond.get("integrator"), bool(cond.get("x64")), episodes)
        groups.setdefault(key, []).append(dict(
            name=cname, shift=cond.get("shift") or {},
            no_randomization=bool(cond.get("no_randomization"))))
    (run_dir / "eval").mkdir(exist_ok=True)
    for i, ((integrator, x64, episodes), conds) in enumerate(groups.items()):
        spec = run_dir / "eval" / f"_conditions_{i}.json"
        spec.write_text(json.dumps(conds))
        cmd = [py, "scripts/evaluate.py", "--run-dir", str(run_dir), "--conditions", str(spec),
               "--no-plot", "--episodes", str(episodes), "--policy", "pd" if job["pd"] else "ppo"]
        if integrator:
            cmd += ["--integrator", integrator]
        if x64:
            cmd.append("--x64")
        with open(run_dir / "eval.log", "a") as log:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        spec.unlink(missing_ok=True)
        if rc:
            return f"FAILED eval {[c['name'] for c in conds]} for {run_dir} (see eval.log)"
    return f"done {job['variant']} seed {job['seed']} in {time.time() - t0:.0f} s"


def cmd_run(exp: dict, out: Path, args) -> None:
    jobs = list(jobs_of(exp, out))
    n_train = sum(1 for j in jobs if not j["pd"])
    print(f"{exp['name']}: {len(exp['_variants'])} variants, {len(jobs)} jobs "
          f"({n_train} trainings), {len(exp['eval']['conditions'])} eval conditions -> {out}")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['variant']:40s} seed {j['seed']}  {j['overrides']}")
        return
    check_resolved(exp, out, args.force)

    gpus = args.gpus.split(",") if args.gpus else [None]
    n_workers = args.jobs or len(gpus)
    slots: queue.Queue = queue.Queue()
    for i in range(n_workers):
        slots.put(gpus[i % len(gpus)])

    def work(job):
        gpu = slots.get()
        try:
            return run_job(job, exp, gpu, args)
        finally:
            slots.put(gpu)

    with cf.ThreadPoolExecutor(n_workers) as pool:
        for i, msg in enumerate(pool.map(work, jobs), 1):
            print(f"[{i}/{len(jobs)}] {msg}", flush=True)
    cmd_summarize(exp, out)


# ---------------------------------------------------------------------- summary


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def cmd_summarize(exp: dict, out: Path) -> None:
    metrics = ("rms_err_mm_mean", "rms_err_mm_std", "steady_state_err_mm_mean",
               "termination_rate", "V_mean", "clf_violation_rate", "mean_abs_torque")
    rows = []
    for vname in exp["_variants"]:
        for cname in exp["eval"]["conditions"]:
            for mode in MODES:
                vals = {m: [] for m in metrics}
                for f in sorted((out / vname).glob(f"seed*/eval/{cname}.json")):
                    r = json.loads(f.read_text()).get(mode)
                    if r:
                        for m in metrics:
                            vals[m].append(r[m])
                if not vals["rms_err_mm_mean"]:
                    continue
                row: dict[str, Any] = dict(variant=vname, condition=cname, mode=mode,
                                           seeds=len(vals["rms_err_mm_mean"]))
                for m in metrics:
                    row[m], row[m + "__sd"] = _mean_std(vals[m])
                rows.append(row)
    if not rows:
        print("no results yet")
        return

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- ranking by the objective
    ob = exp["objective"]
    pen = ob.get("termination_penalty_mm", 10.0)
    score = {}
    for v in exp["_variants"]:
        rs = [r for r in rows if r["variant"] == v and r["condition"] == ob["condition"]]
        if len(rs) == len(MODES):
            score[v] = statistics.fmean(r[ob["metric"]] + pen * r["termination_rate"] for r in rs)
    ranked = sorted(score, key=lambda v: score[v])

    lines = [f"# {exp['name']}", "",
             f"Objective: mean over {', '.join(MODES)} of `{ob['metric']}` + {pen} x termination rate, "
             f"condition `{ob['condition']}` (lower is better). Values are mean ± std over seeds.", ""]
    conds = list(exp["eval"]["conditions"])
    for mode in MODES:
        lines += [f"## {mode}: RMS tracking error [mm]", "",
                  "| rank | variant | score | " + " | ".join(conds) + " |",
                  "|---|---|---|" + "---|" * len(conds)]
        for i, v in enumerate(ranked + [v for v in exp["_variants"] if v not in score], 1):
            cells = []
            for c in conds:
                r = next((r for r in rows if (r["variant"], r["condition"], r["mode"]) == (v, c, mode)), None)
                if r is None:
                    cells.append("–")
                else:
                    t = f" ⚠{r['termination_rate']:.2f}" if r["termination_rate"] > 0 else ""
                    cells.append(f"{r['rms_err_mm_mean']:.2f} ± {r['rms_err_mm_mean__sd']:.2f}{t}")
            sc = f"{score[v]:.3f}" if v in score else "–"
            lines.append(f"| {i if v in score else '–'} | {v} | {sc} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["⚠x = termination rate. Full metrics (settled error, V, CLF-violation rate, torque, "
              "episode spread) are in summary.csv.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if ranked:
        print(f"best: {ranked[0]}  overrides {exp['_variants'][ranked[0]]}")


# ------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "summarize"))
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--name", help="override the experiment name (output directory)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra base override for every variant, e.g. --set reward.clf.sigma_v=0.1")
    ap.add_argument("--seeds", help="comma-separated seeds, overrides the YAML")
    ap.add_argument("--gpus", help="comma-separated GPU ids; one job per GPU at a time")
    ap.add_argument("--jobs", type=int, help="parallel jobs (default: number of GPUs, else 1)")
    ap.add_argument("--ppo-config", help="override the PPO config (e.g. configs/ppo_cpu_demo.yaml)")
    ap.add_argument("--total-timesteps", type=float)
    ap.add_argument("--episodes", type=int, help="override eval episodes for every condition")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="reuse an output dir with a changed definition")
    args = ap.parse_args()
    if args.total_timesteps:
        args.total_timesteps = int(args.total_timesteps)

    cli_sets = {}
    for item in args.set:
        k, _, v = item.partition("=")
        cli_sets[k] = yaml.safe_load(v)
    exp = load_experiment(args.experiment, cli_sets)
    if args.name:
        exp["name"] = args.name
    if args.seeds:
        exp["seeds"] = [int(s) for s in args.seeds.split(",")]
    out = ROOT / "runs" / exp["name"]
    if args.command == "run":
        cmd_run(exp, out, args)
    else:
        cmd_summarize(exp, out)


if __name__ == "__main__":
    main()
