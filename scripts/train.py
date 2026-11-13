"""Train a PPO policy on the gantry environment.

    python scripts/train.py --env-config configs/gantry.yaml --ppo-config configs/ppo.yaml \
        --run-dir runs/demo [--total-timesteps 2e6] [--num-envs 256]

The run directory receives copies of both configs, `metrics.csv`, and
`models.eqx` (actor, critic, observation normalisers).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import jax
import yaml

from gantry_sim import GantryEnv
from gantry_sim.params import apply_overrides, env_config_from_dict, gantry_params_from_dict, load_yaml
from rl.ppo import load_ppo_config, make_trainer, save_models


REWARD_PRESETS = {
    "clf": {"reward.mode": "clf"},
    "clf_tracking_only": {"reward.mode": "clf", "reward.clf.w_vdot": 0.0},  # paper ablation
    "heuristic": {"reward.mode": "heuristic"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default="configs/gantry.yaml")
    ap.add_argument("--ppo-config", default="configs/ppo.yaml")
    ap.add_argument("--run-dir", default="runs/demo")
    ap.add_argument("--total-timesteps", type=float, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--reward", choices=tuple(REWARD_PRESETS), default=None,
                    help="reward preset (clf_tracking_only sets w_vdot = 0, the paper's ablation)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an env-config entry, e.g. --set reward.clf.sigma_vdot=25")
    args = ap.parse_args()

    env_yaml = load_yaml(args.env_config)
    sets = dict(REWARD_PRESETS[args.reward]) if args.reward else {}
    for item in args.set:
        key, _, value = item.partition("=")
        sets[key] = yaml.safe_load(value)
    env_yaml = apply_overrides(env_yaml, sets)
    params, env_cfg = gantry_params_from_dict(env_yaml["model"]), env_config_from_dict(env_yaml)
    cfg = load_ppo_config(args.ppo_config)
    overrides = {k: v for k, v in dict(
        total_timesteps=None if args.total_timesteps is None else int(args.total_timesteps),
        num_envs=args.num_envs, seed=args.seed).items() if v is not None}
    cfg = cfg.replace(**overrides)

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "gantry.yaml", "w") as f:  # exactly the env config that was trained
        yaml.safe_dump(env_yaml, f, sort_keys=False)
    shutil.copy(args.ppo_config, run / "ppo.yaml")
    if overrides:
        (run / "overrides.txt").write_text("".join(f"{k}: {v}\n" for k, v in overrides.items()))

    env = GantryEnv(params, env_cfg)
    print(f"devices: {jax.devices()}")
    print(f"actor obs {env.actor_obs_size}, critic obs {env.critic_obs_size}, "
          f"episode {env_cfg.episode_steps} steps @ {1 / env_cfg.control_dt:.0f} Hz, "
          f"integrator {env_cfg.integrator.method}, reward {env_cfg.reward.mode}"
          + (f" (w_v {env_cfg.reward.clf.w_v}, w_vdot {env_cfg.reward.clf.w_vdot})"
             if env_cfg.reward.mode == "clf" else ""))
    print(f"{cfg.num_updates} updates x {cfg.batch_size} steps = {cfg.num_updates * cfg.batch_size:,} env steps")

    init_fn, update_fn = make_trainer(env, cfg)
    ts, runner = init_fn(jax.random.PRNGKey(cfg.seed))

    t0 = time.time()
    with open(run / "metrics.csv", "w", newline="") as f:
        writer = None
        for i in range(1, cfg.num_updates + 1):
            ts, runner, metrics = update_fn(ts, runner)
            if i % cfg.log_every == 0 or i == 1 or i == cfg.num_updates:
                m = {k: float(v) for k, v in metrics.items()}
                elapsed = time.time() - t0
                steps = i * cfg.batch_size
                row = dict(update=i, env_steps=steps, wall_s=round(elapsed, 1), **m)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(f"[{i:5d}/{cfg.num_updates}] {steps:>11,} steps  {steps / elapsed:>9,.0f} sps | "
                      f"return {m['episode_return']:8.1f}  rms sp {m['rms_err_setpoint_mm']:6.2f} mm  "
                      f"tr {m['rms_err_trajectory_mm']:6.2f} mm  term {m['termination_rate']:.3f}  "
                      f"V/sig {m['V_over_sigma']:.3f}  viol {m['clf_violation_rate']:.3f}  "
                      f"std {m['action_std']:.3f}  kl {m['approx_kl']:.4f}")
            if i % cfg.checkpoint_every == 0:
                save_models(run / "models.eqx", ts.models)

    save_models(run / "models.eqx", ts.models)
    print(f"done in {time.time() - t0:.0f} s -> {run / 'models.eqx'}")


if __name__ == "__main__":
    main()
