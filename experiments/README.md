# Experiments

All experiments use `scripts/experiment.py`. It expands an experiment YAML into
(variant × seed) jobs. Each job trains, then evaluates every test condition, and
the script aggregates the results over seeds. Results go to `runs/<name>/`:
`summary.md` holds a ranked table (mean ± std over seeds) and `summary.csv` holds every metric.

```bash
python scripts/experiment.py run <exp.yaml> --dry-run           # list jobs
python scripts/experiment.py run <exp.yaml> --gpus 0,1,2,3      # one job per GPU
python scripts/experiment.py summarize <exp.yaml>               # re-aggregate
# quick CPU check of the plumbing (results meaningless):
python scripts/experiment.py run <exp.yaml> --name smoke --seeds 0 \
    --ppo-config configs/ppo_cpu_demo.yaml --total-timesteps 40000 --episodes 8
```

Jobs that already have results are skipped, so an interrupted run can simply be restarted.
Variant names come from the grid values, e.g. `sigma_vdot=25,decay_rate=5`. `formula` means `null`,
i.e. the paper's expression is used.

## 1. Scale grid search (`scale_grid_stage1.yaml`, `scale_grid_stage2.yaml`)

CLF-RL has two reward scales, σ_v (tracking) and σ_V̇ (decay), plus the decay rate λ.
The search runs in two stages:

| Stage | Searches | Fixed | Variants × seeds |
|---|---|---|---|
| 1 | σ_v ∈ {0.025, 0.05, 0.1, 0.2, 0.4, 0.8, formula} | w_V̇ = 0 (decay off) | 7 × 3 = 21 |
| 2 | σ_V̇ ∈ {5, 10, 25, 50, 100, formula} × λ ∈ {2, 5, 10} s⁻¹, plus a tracking-only control | σ_v = stage-1 winner | 19 × 3 = 57 |

- **Why two stages.** σ_v changes r_v regardless of the decay term, so it is tuned first with the decay term off. σ_V̇ and λ only matter once σ_v is set. A full 7×6×3 grid would be 378 trainings.
- **Selection.** Variants are ranked on task performance, not reward, because the scales change the reward itself. The score is the in-distribution RMS tracking error averaged over setpoint and trajectory, plus 10 mm × termination rate.
- **Robustness column.** A `payload` condition (+0.25 kg on the carriage, not seen in training) is reported next to the score. It flags scales that win in-distribution but are brittle.

After stage 1, pass the winner into stage 2:
```bash
python scripts/experiment.py run experiments/scale_grid_stage2.yaml \
    --set reward.clf.sigma_v=0.1 --name scale_grid_stage2_sv0.1 --gpus 0,1,2,3
```
Then put the final σ_v, σ_V̇ and λ into `configs/gantry.yaml` before the ablation.

### How the current defaults were chosen (before this grid)
1. **Paper formulas.** I started from σ_v = μ_max(P)·η_max² and σ_V̇ = 2‖P‖η_max·η̇_max + λμ_max·η_max², with η_max = 5 mm and η̇_max = 0.5 picked by hand. That gives σ_v = 1.35 and σ_V̇ = 284.
2. **Diagnosis.** Trained policies reached V/σ_v ≈ 0.08, so r_v sat at about 9.2 of 10 and its slope at their operating point was nearly flat. The decay term's argument (V̇ + λV)/σ_V̇ was on the order of 1/283, so r_V̇ was effectively zero. CLF and tracking-only policies were indistinguishable, and both trailed the heuristic reward.
3. **Measure.** I ran the PD baseline for 256 randomized episodes and logged η, η̇, V and V̇:
   - |e_p|: p95 3.8 mm
   - |e_v|: p95 0.08 m/s
   - V: p50 0.05, p95 1.09
   - |V̇|: p50 0.35, p95 23

   Applied literally to ‖η‖ and ‖η̇‖, the paper's "empirical bounds" would give σ_v ≈ 334 and σ_V̇ ≈ 2.7·10⁴. P weights position about 900× more than velocity, so ‖η‖ is dominated by velocity error and the formula bounds V very loosely.
4. **Calibrate on V and V̇ directly.** I set σ_V̇ = 25 (≈ p95 |V̇|), so the penalty can saturate. This gave a consistent 15–20% gain over tracking-only on 2 seeds while σ_v was still loose.
5. **Tighten σ_v.** I set σ_v = 0.2, a ~1.9 mm position-equivalent error, so r_v keeps a usable slope at the 0.5–1 mm errors trained policies reach. This brought CLF-RL level with, or ahead of, the heuristic. With σ_v this tight, the decay term's benefit was no longer distinguishable from seed noise.

Step 5 was a single guess checked on 2 seeds; it was not a search. The grid above replaces it.

## 2. Ablation (`ablation.yaml`)

Six variants × 5 seeds, following the paper's Sec. IV:

| Variant | Question it answers |
|---|---|
| `pd_baseline` | classical reference (deterministic, 1 run) |
| `heuristic` | hand-shaped tracking reward (the paper's "Custom" baseline) |
| `clf_tracking_only` | r_v only (paper ablation, w_V̇ = 0) |
| `clf` | full CLF-RL with the tuned scales |
| `clf_paper_scales` | CLF-RL with the paper's σ formulas: does calibration matter? |
| `clf_command_obs` | actor sees the raw command, not the planned reference (paper: reference only in the reward) |

Every policy is tested under 8 conditions. Plant changes affect the *true* plant only; the planner and the actor's encoder kinematics stay nominal.

| Condition | Change | In training? |
|---|---|---|
| `in_dist` | training randomization | yes |
| `nominal` | no randomization | yes |
| `payload` / `heavy_payload` | +0.25 / +0.5 kg on the carriage (m_x ×1.5/×2.0, m_y ×1.21/×1.42) | no (masses ±20%) |
| `soft_belts` | κ ×0.6 | no (±30%) |
| `high_friction` | carriage F_c, F_s ×2 | no (±50%) |
| `radius_error` | r_A ×1.01, r_B ×0.99 (kinematic calibration error) | no |
| `hifi_solver` | Kvaerno5, float64 | sim-to-sim check |

**Metrics.** For each variant and condition the summary reports:
- RMS tracking error vs. the planned reference;
- settled error vs. the goal;
- termination rate;
- mean raw V (comparable across σ settings);
- CLF-violation rate;
- torque use;
- per-episode spread.

All are given as mean ± std over seeds. The paper's main robustness claim is *lower variance under perturbations*. That corresponds to the std columns and to `rms_err_mm_std` (spread across perturbed episodes).
