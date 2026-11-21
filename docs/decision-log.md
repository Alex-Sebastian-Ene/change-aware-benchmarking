# Decision log

One entry per decision. Newest at the bottom.

| Date | Decision | Why | Who |
|------|----------|-----|-----|
|      |          |     |     |
| 2026-10-25 | Extend the fixed IRT protocol through gp-IRT, 1D Fisher, the D={1,2,3,5,8} random-selection sweep, and static/adaptive V-optimal MIRT. | Isolate estimator, selector and dimension changes; reuse existing gp-IRT, tune on validation, retain per-model paired results, and leave baseline defaults unchanged. Cost-awareness and fine-tuning analysis remain deferred. | A. Vijaykrishnan |
| 2026-10-26 | Separate frozen-model design from the fitting backend, expose parameter/callback adapters in stages 05–07, and add a fit-free visualisation notebook 08. | Teammates can reuse fitted IRT and ability estimators, attach V-optimal selection without refitting, and compare selection/estimation using the same records. | A. Vijaykrishnan |
