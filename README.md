# Attendance Disclosure Study

Does announcing phase-specific attendance-check quotas change attendance when the instructor uses exactly the same checks? This project models a 16-lesson course with 10 checks. It compares announcing only the term total with also announcing the allocation across the two eight-lesson phases.

The study uses synthetic schedules. It computes exact student best responses by backward induction over 7,056 feasible schedules and compares them with tabular Q-learning. Under the first-warning rule, disclosing the quotas changes expected attendance by **−3.92 percentage points** at an attendance cost of 0.575 and **+7.91 points** at a cost of 0.650. The corresponding held-out Q-learning estimates are **−4.53** and **+8.03** points.

## Reproduce the experiments

Use Python 3.12. All experiments run on CPU.

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python attendance_study.py
python build_report_assets.py
```

For a quick check, run `python attendance_study.py --quick`. It runs the exact analysis and a two-seed learning experiment and writes to `quick_results/`. Generating the report figures requires the full experiment in `results/`.

## Model and evaluation

- The phase quotas are sampled uniformly from `4+6`, `5+5`, and `6+4`. Within each phase, check dates are sampled uniformly without replacement. Exact averages assign equal probability to each quota pair and equal weight to schedules within each pair.
- The student chooses before seeing whether the current lesson is checked. After the choice, the check outcome becomes public. The linear benchmark penalizes every detected absence. The main experiment gives a warning for the first detected absence and penalizes each subsequent detection.
- The learning experiment uses seven costs, 16 training seeds per condition, and 32,768 semesters per seed. Both disclosure conditions share the schedule draws. Frozen policies are evaluated on the same 4,096 held-out semesters. The paired confidence intervals summarize variation across training seeds.

## Project files

| File | Purpose |
| --- | --- |
| `attendance_study.py` | Schedule generation, exact planning, Q-learning, and evaluation |
| `build_report_assets.py` | Vector figures and LaTeX-ready tables generated from the full results |
| `tests/test_attendance_study.py` | Checks schedule weights, Bayesian beliefs, information visibility, penalties, and exact benchmarks |

The full run writes `results/summary.json` for exact and learned outcomes, `results/training_curve.csv` for validation checkpoints, `results/test_by_seed.csv` for held-out metrics, and `results/raw_results.npz` for schedules, learned Q tables, and rollout records. To inspect the raw archive, use `numpy.load` with its default `allow_pickle=False`. The `state_table` columns are lesson index, observed first-phase checks, observed second-phase checks, detected absences, and disclosed first-phase quota. A zero in the last column means the quota was undisclosed. Q-table actions `0` and `1` represent attendance and absence. In `test_attendance_mask`, the low bit corresponds to lesson 1. Figures and tables are emitted to `figures/`.
