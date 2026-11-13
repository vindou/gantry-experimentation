"""Test a trained policy (or the PD baseline) on held-out episodes.

    python scripts/evaluate.py --run-dir runs/demo                      # PPO, training integrator
    python scripts/evaluate.py --run-dir runs/demo --integrator kvaerno5 # high-fidelity solver
    python scripts/evaluate.py --run-dir runs/demo --policy pd           # classical baseline
    python scripts/evaluate.py --run-dir runs/demo --no-randomization    # nominal plant only

Per task mode it reports tracking error (from the true carriage position),
termination rate, and torque use; writes `eval/<tag>.json` and a plot of
episode 0 to `eval/<tag>_<mode>.png` inside the run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gantry_sim import SETPOINT, TRAJECTORY, GantryEnv, load_config, make_pd_policy  # noqa: E402
from gantry_sim.params import IntegratorConfig  # noqa: E402
from gantry_sim.randomization import make_shift  # noqa: E402
from rl.ppo import load_models, load_ppo_config, make_actor_policy  # noqa: E402

MODES = {"setpoint": SETPOINT, "trajectory": TRAJECTORY}


def rollout_fn(env: GantryEnv, policy):
    """Deterministic episodes; records until each episode ends.

    Returns jitted f(keys, mode, shift, rand_scale): task mode, plant shift and
    randomisation scale are runtime arguments, so all conditions share one compile.
    """
    T = env.cfg.episode_steps

    def run(key, mode, shift, rand_scale):
        s, ao, _ = env.reset(key, task_mode=mode, shift=shift, rand_scale=rand_scale)

        def body(carry, k):
            s, ao, alive = carry
            a = policy(ao)
            s, ao, _, r, _, info = env.step_no_reset(k, s, a)
            # a step counts only if the episode is still running AFTER it, so the
            # terminating step (state may be non-finite) never enters the metrics
            alive = alive & ~info["terminated"]
            rec = dict(p=info["p"], p_ref=info["p_ref"], p_cmd=info["p_cmd"], err=info["pos_err"],
                       u=info["u"], TA=info["TA"], TB=info["TB"], V=info["V_norm"], Vraw=info["V"],
                       viol=info["clf_violation"], reward=r)
            rec = jax.tree.map(lambda x: jnp.where(alive, jnp.nan_to_num(x), jnp.zeros_like(x)), rec)
            rec["alive"] = alive
            return (s, ao, alive), rec

        _, rec = jax.lax.scan(body, (s, ao, jnp.asarray(True)), jax.random.split(key, T))
        return rec

    return jax.jit(jax.vmap(run, in_axes=(0, None, None, None)))


def summarize(rec: dict, env: GantryEnv) -> dict:
    alive = rec["alive"].astype(jnp.float32)
    n_alive = jnp.maximum(alive.sum(1), 1.0)
    rms = jnp.sqrt((rec["err"] ** 2 * alive).sum(1) / n_alive)
    terminated = alive[:, -1] == 0

    goal_err = jnp.linalg.norm(rec["p_cmd"] - rec["p"], axis=-1)
    # steady-state window: the last 100 ms of each setpoint hold
    # integer step arithmetic: the goal switches exactly at multiples of hold_steps
    steps = jnp.arange(env.cfg.episode_steps) + 1
    hold_steps = round(env.cfg.task.hold_time / env.cfg.control_dt)
    window = round(0.1 / env.cfg.control_dt)
    ss_mask = jnp.mod(steps, hold_steps) >= hold_steps - window
    ss = alive * ss_mask[None]
    ss_err = (goal_err * ss).sum(1) / jnp.maximum(ss.sum(1), 1.0)

    f = lambda x: float(x)
    return dict(
        episodes=int(alive.shape[0]),
        rms_err_mm_mean=f(1e3 * rms.mean()),
        rms_err_mm_p95=f(1e3 * jnp.percentile(rms, 95)),
        steady_state_err_mm_mean=f(1e3 * ss_err.mean()),
        max_err_mm=f(1e3 * (rec["err"] * alive).max()),
        termination_rate=f(terminated.mean()),
        mean_abs_torque=f((jnp.abs(rec["u"]).mean(-1) * alive).sum() / alive.sum()),
        return_mean=f((rec["reward"] * alive).sum(1).mean()),
        rms_err_mm_std=f(1e3 * rms.std()),
        V_over_sigma_mean=f((rec["V"] * alive).sum() / alive.sum()),
        V_mean=f((rec["Vraw"] * alive).sum() / alive.sum()),  # comparable across sigma_v settings
        clf_violation_rate=f((rec["viol"] * alive).sum() / alive.sum()),
    )


def plot_episode(rec: dict, env: GantryEnv, title: str, path: Path) -> None:
    T = env.cfg.episode_steps
    t = (jnp.arange(T) + 1) * env.cfg.control_dt
    i = 0
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for j, (name, c) in enumerate((("x", "C0"), ("y", "C1"))):
        ax[0].plot(t, 1e3 * rec["p_cmd"][i, :, j], ":", color=c, alpha=0.6, label=f"{name} cmd")
        ax[0].plot(t, 1e3 * rec["p_ref"][i, :, j], "--", color=c, label=f"{name} ref")
        ax[0].plot(t, 1e3 * rec["p"][i, :, j], color=c, label=name)
    ax[0].set_ylabel("position [mm]")
    ax[0].legend(ncol=6, fontsize=8)
    ax[1].semilogy(t, 1e3 * rec["err"][i] + 1e-6, label="|p_ref - p| [mm]")
    ax[1].semilogy(t, rec["V"][i] + 1e-9, label=r"$V/\sigma_v$")
    ax[1].set_ylabel("tracking error")
    ax[1].legend(fontsize=8)
    ax[2].plot(t, rec["u"][i, :, 0], label=r"$\tau_A$")
    ax[2].plot(t, rec["u"][i, :, 1], label=r"$\tau_B$")
    ax[2].set_ylabel("torque [N m]")
    ax[2].legend(fontsize=8)
    ax[3].plot(t, rec["TA"][i], label="$T_A$")
    ax[3].plot(t, rec["TB"][i], label="$T_B$")
    ax[3].set_ylabel("belt force [N]")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/demo")
    ap.add_argument("--policy", choices=("ppo", "pd"), default="ppo")
    ap.add_argument("--integrator", choices=("rk4", "kvaerno5", "tsit5"), default=None,
                    help="override the integrator (default: as in the run's gantry.yaml)")
    ap.add_argument("--mode", choices=("setpoint", "trajectory", "both"), default="both")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345, help="use a seed not used in training")
    ap.add_argument("--x64", action="store_true", help="float64 (recommended with kvaerno5)")
    # one condition from flags ...
    ap.add_argument("--no-randomization", action="store_true")
    ap.add_argument("--shift", action="append", default=[], metavar="PATH=FACTOR",
                    help="out-of-distribution plant change, e.g. --shift m_x=1.5 (true plant only)")
    ap.add_argument("--tag", default=None, help="name for the output files")
    # ... or many from a file (one compilation for all of them)
    ap.add_argument("--conditions", type=Path,
                    help='JSON list of {"name", "shift": {path: factor}, "no_randomization"}')
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)

    run = Path(args.run_dir)
    params, env_cfg = load_config(run / "gantry.yaml")
    train_env = GantryEnv(params, env_cfg)
    if args.integrator:
        ic = env_cfg.integrator
        env_cfg = env_cfg.replace(integrator=IntegratorConfig(
            method=args.integrator, substeps=ic.substeps, rtol=ic.rtol, atol=ic.atol,
            max_steps=ic.max_steps))
    env = GantryEnv(params, env_cfg)

    if args.conditions:
        conditions = json.loads(args.conditions.read_text())
    else:
        shift = {}
        for item in args.shift:
            path, _, factor = item.partition("=")
            shift[path] = float(factor)
        tag = args.tag or (f"{args.policy}_{env_cfg.integrator.method}"
                           + ("_nominal" if args.no_randomization else ""))
        conditions = [dict(name=tag, shift=shift, no_randomization=args.no_randomization)]

    if args.policy == "ppo":
        models = load_models(run / "models.eqx", train_env, load_ppo_config(run / "ppo.yaml"))
        policy = make_actor_policy(models)
    else:
        policy = make_pd_policy(env)
    rollout = rollout_fn(env, policy)

    out_dir = run / "eval"
    out_dir.mkdir(exist_ok=True)
    modes = list(MODES) if args.mode == "both" else [args.mode]
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    for cond in conditions:
        shift = make_shift(params, cond.get("shift"))
        rand_scale = 0.0 if cond.get("no_randomization") else 1.0
        results = {}
        for mode in modes:
            rec = rollout(keys, jnp.asarray(MODES[mode], jnp.int32), shift, jnp.asarray(rand_scale))
            results[mode] = r = summarize(rec, env)
            if not args.no_plot:
                plot_episode(rec, env, f"{cond['name']} - {mode}", out_dir / f"{cond['name']}_{mode}.png")
            print(f"{cond['name']:>24s} {mode:>10s}: rms {r['rms_err_mm_mean']:7.2f} mm "
                  f"(p95 {r['rms_err_mm_p95']:7.2f})  settled {r['steady_state_err_mm_mean']:6.2f} mm  "
                  f"term {r['termination_rate']:.3f}  V/sig {r['V_over_sigma_mean']:.3f}  "
                  f"viol {r['clf_violation_rate']:.3f}  |tau| {r['mean_abs_torque']:.3f}")
        (out_dir / f"{cond['name']}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
