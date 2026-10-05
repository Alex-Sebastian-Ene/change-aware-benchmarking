# Change-Aware Efficient Benchmarking of Language Models

ELEC6200 MEng Group Design Project, Group 7.
Supervisor: Dr Leonardo Aniello. External partner: Til Jordan (Bloctopus / Consyn).

The idea: estimate a model's benchmark score from a small, well-chosen subset of items
(using IRT), detect real capability changes between fine-tuning checkpoints, and give
calibrated 80% / 95% intervals for the score and for the change.

## Setup

You need [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

`uv sync` creates `.venv/` and installs everything from `uv.lock`, so everyone gets the same versions.

## Everyday uv commands

| Task | Command |
|------|---------|
| Add a dependency | `uv add <package>` |
| Add a dev-only dependency | `uv add --dev <package>` |
| Remove a dependency | `uv remove <package>` |
| Run a script | `uv run python scripts/<name>.py` |
| Run tests | `uv run pytest` |
| Lint / format | `uv run ruff check .` / `uv run ruff format .` |
| Jupyter kernel | pick `.venv` as the kernel in VS Code |

Commit `pyproject.toml` and `uv.lock` together whenever you add or remove a package.

## Layout

```
src/cabench/        the package, one module per core objective
  data.py           response matrix (models x items, 0/1)        J. Hall
  baselines.py      noise reference + random subsets             S. Ene
  irt.py            IRT fit, static + adaptive subsets           A. Vijaykrishnan, A. Chitnis
  intervals.py      prediction intervals + coverage              S. Ene
  change.py         change detection across checkpoints          S. Ene
  probe.py          activation probe vs IRT                      A. Vijaykrishnan, A. Chitnis
  splits.py         split definitions                            S. Ene, J. Hall
scripts/            entry points that run experiments
notebooks/          exploration only - move anything reusable into src/
tests/              pytest tests
data/raw/           downloaded data (git-ignored)
data/processed/     derived data, e.g. response matrices (git-ignored)
results/            outputs and figures (git-ignored)
docs/               decision log, PM handover template
```

## Suggested build order

Start with the simplest thing that works end to end, then make it better.

1. **Response matrix.** Get one benchmark into a pandas DataFrame: rows = models, columns = items, values = 0/1.
   Everything else reads this.
2. **Baselines.** Full-benchmark accuracy, the noise reference (same model run several times),
   and random subsets of size k. These are what IRT has to beat.
3. **IRT.** Fit a 1PL (Rasch) model first, then 2PL. Pick a static subset of items and compare
   its score error against a random subset of the same size.
4. **Intervals.** Put 80% / 95% intervals on the predicted score and check how often the full-eval
   score falls inside them.
5. **Change detection.** Fix an anchor item set, score two checkpoints on it, and test whether the
   difference is larger than delta and above the noise reference.
6. Adaptive selection, the probe, and distribution-shift splits come after this.

## Working together

- Branch per task (`feature/irt-1pl`, `fix/loader-nan`), open a PR, one review before merge.
- Keep `main` passing: `uv run ruff check . && uv run pytest`.
- Fix random seeds and record them with every result.
- Do not commit data, model weights or API keys. Put secrets in `.env` (git-ignored).
- Record decisions in `docs/decision-log.md`. The PM role rotates every 3 weeks; use
  `docs/handover-template.md` at each handover.

## Contingencies

| If | Then |
|----|------|
| Cloud delayed | Use Iridis |
| Changes below noise | Anchor items, paired tests |
| Shift hurts predictions | Report extrapolation split |
| Probe fails | IRT stays primary |
| Time runs short | Drop stretch objectives first |
