# Generate_Data_v2.4 - Small run report

Summary:
- Script: `Generate_Data_v2.4_runner.py` run on 2026-02-16.
- Small-run settings: `n_subjects=50`, `trials_per_sub=60` (total 3000 trials).

Files produced:
- Data CSV: `2_Data/Generate_Data/gp_ddm_v2.4_small.csv`
- Figures (in `3_Figures/Generate_Data_v2.4_checks`):
  - `RT_distribution_v2.4.png` — overall RT histogram/kde
  - `RT_by_label_v2.4.png` — RT by `label` (self/stranger)
  - `SPE_dist_v2.4.png` — signed prediction error distribution

Quick metrics (from script stdout):
- RT mean: ~0.738 s; median: ~0.614 s
- Percent RT ≤ 2s: ~98.97%
- Proportion upper (label=1): ~81.07%; lower (label=2): ~13.77%
- Mean SPE: ~-277.44 ms; SD SPE: ~144.07 ms; Cohen's d (paired): ~-1.93
- GP predictive std (v): mean ~0.2585

Next suggested steps:
1. Inspect `gp_ddm_v2.4_small.csv` in a spreadsheet or Python to verify trial-level parameters and RT/responses.
2. Review the saved figures for Layer 1 checks (RT & SPE). If acceptable, run parameter-recovery experiments (Layer 3) by increasing `n_subjects` and adding noise sweeps.

If you want, I can:
- Open and display any of the PNGs inline here.
- Run a parameter recovery sweep (specify ranges/noise levels).
- Export a smaller sample CSV or aggregate summaries.

