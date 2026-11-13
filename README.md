# Gantry RL simulator

JAX simulator of the belt-driven gantry in [model.tex](model.tex), wrapped as a
vectorised RL environment and trained with **CLF-RL** (Li, Olkin, Yue, Ames,
RA-L 2025, arXiv:2508.09354) using asymmetric-actor-critic PPO.

The training formulation is written out in the "CLF-RL training formulation" section of [model.tex](model.tex):
* **Reference generation:** a reduced-order rigid-belt planner (the gantry's counterpart to the paper's H-LIP model).
* **CLF:** V = ηᵀPη, with P from the CARE of the double integrator.
* **Rewards:** CLF tracking, CLF decay, and regularization.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
# Linux + NVIDIA GPU (e.g. RTX 4090), additionally:
pip install -r requirements-gpu.txt
```

## Train and test

```bash
# quick CPU demo (~2 min on a laptop); --reward picks clf | clf_tracking_only | heuristic
python scripts/train.py --ppo-config configs/ppo_cpu_demo.yaml --run-dir runs/clf --reward clf
# full config (intended for GPU)
python scripts/train.py --ppo-config configs/ppo.yaml --run-dir runs/gpu --num-envs 4096

python scripts/evaluate.py --run-dir runs/demo                              # PPO, training integrator
python scripts/evaluate.py --run-dir runs/demo --integrator kvaerno5 --x64  # high-fidelity solver
python scripts/evaluate.py --run-dir runs/demo --policy pd                  # PD baseline
python scripts/evaluate.py --run-dir runs/demo --no-randomization           # nominal plant
pytest
```

Evaluation writes `runs/<name>/eval/<tag>.json` and per-mode plots.

The CLF scale grid search and the ablation study are run through `scripts/experiment.py`; see [experiments/README.md](experiments/README.md).

## Layout

| Path | Contents |
|---|---|
| `configs/gantry.yaml` | plant parameters, env, tasks, observations, reward, domain randomisation |
| `configs/ppo*.yaml` | PPO hyper-parameters |
| `gantry_sim/params.py` | YAML → `GantryParams` (pytree) and `EnvConfig` (static) |
| `gantry_sim/dynamics.py` | friction, belt stiffness/forces, `vector_field(z, u, p)` — the EOMs |
| `gantry_sim/integrator.py` | one control period with zero-order-hold torque: `rk4` (training) or Diffrax `kvaerno5`/`tsit5` (validation) |
| `gantry_sim/planner.py` | rigid-belt inverse dynamics; feasible minimum-jerk durations and trajectory scaling |
| `gantry_sim/clf.py` | CARE → P, V(η), σ_v, σ_V̇, CLF tracking/decay rewards |
| `gantry_sim/tasks.py` | setpoint sequences and sum-of-sines trajectories, planned references |
| `gantry_sim/randomization.py` | per-episode multiplicative parameter randomisation |
| `gantry_sim/env.py` | `GantryEnv`: `reset`, `step` (auto-reset), `step_no_reset` |
| `gantry_sim/baselines.py` | Cartesian PD controller on encoder data |
| `rl/networks.py`, `rl/ppo.py` | actor/critic MLPs, obs normaliser, PPO |
| `scripts/` | `train.py`, `evaluate.py` |

## Information structure

* **Actor (deployed):** stacked history of encoder-derived quantities (position
  via nominal rigid-belt kinematics, finite-difference velocity, with noise and
  quantisation), the reference and its preview, and the previous action.
* **Critic (training only):** actor inputs + true carriage state, belt stretch,
  episode time, and the episode's randomised parameters.
* **Reward:** from the true carriage position/velocity.
