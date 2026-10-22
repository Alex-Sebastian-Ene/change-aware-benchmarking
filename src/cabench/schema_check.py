"""Schema check run on every new batch (acceptance test 3 in the matrix spec).

Passes when:
  1. zero duplicate (run_id, item_id) keys in the long table
  2. row count per run equals the item count of the run's config
  3. every item_id, model and config_id resolves in its table

Tables, in the data store layout scripts/matrix writes (only these columns are read):
  long    run_id, item_id, model_id, model_revision, config_id, source
          as long/ (one parquet per run, any depth), long.parquet or long.csv
  models  model_id, model_revision, as models.parquet or models.csv
  items   item_id, benchmark, benchmark_version, as items.parquet or items.csv
Configs are the YAML files in configs/harness; a file's hash is its config_id.
Public runs (source == "public") were not run from our configs: their config_id is a
placeholder, so checks 2 and the config_id part of 3 skip them.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from cabench.harness.pins import ConfigError, HarnessConfig, load_config

COLUMNS = {
    "long": ["run_id", "item_id", "model_id", "model_revision", "config_id", "source"],
    "models": ["model_id", "model_revision"],
    "items": ["item_id", "benchmark", "benchmark_version"],
}
MODEL_KEY = ["model_id", "model_revision"]
PUBLIC = "public"
MAX_EXAMPLES = 5


@dataclass
class CheckResult:
    name: str
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def read_tables(directory: Path) -> dict[str, pd.DataFrame]:
    """Read the columns the checks need, as strings."""
    directory = Path(directory)
    tables = {}
    for name, columns in COLUMNS.items():
        folder = directory / name
        parquet, csv = directory / f"{name}.parquet", directory / f"{name}.csv"
        if folder.is_dir() and any(folder.rglob("*.parquet")):
            df = pd.read_parquet(folder)
        elif parquet.exists():
            df = pd.read_parquet(parquet)
        elif csv.exists():
            df = pd.read_csv(
                csv,
                usecols=lambda c, wanted=set(columns): c in wanted,
                dtype=str,
                keep_default_na=False,
                na_values=[""],
            )
        else:
            raise FileNotFoundError(f"no {name}/, {name}.parquet or {name}.csv in {directory}")
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}")
        tables[name] = df[columns].astype("string")
    return tables


def load_configs(directory: Path) -> tuple[dict[str, HarnessConfig], list[str]]:
    """Every config YAML in `directory` keyed by config_id, plus files that failed to load."""
    configs, errors = {}, []
    for path in sorted(Path(directory).glob("*.yaml")):
        try:
            config = load_config(path)
        except (ConfigError, yaml.YAMLError) as e:
            errors.append(f"{path.name}: {e}")
        else:
            configs[config.config_id] = config
    return configs, errors


def _examples(df: pd.DataFrame, columns: list[str]) -> str:
    rows = df[columns].drop_duplicates().head(MAX_EXAMPLES)
    if len(columns) == 1:
        return ", ".join(str(v) for v in rows[columns[0]])
    return ", ".join(str(r) for r in rows.itertuples(index=False, name=None))


def _own_rows(long: pd.DataFrame) -> pd.DataFrame:
    """Rows of runs made from our configs, i.e. every source but public."""
    return long[long["source"].ne(PUBLIC).fillna(True).astype(bool)]


def check_unique_keys(long: pd.DataFrame) -> CheckResult:
    result = CheckResult("zero duplicate (run_id, item_id) keys")
    key = ["run_id", "item_id"]
    dup = long[long.duplicated(key, keep=False)].drop_duplicates(key)
    if len(dup):
        result.failures.append(f"{len(dup)} keys appear more than once, e.g. {_examples(dup, key)}")
    return result


def check_run_counts(
    tables: dict[str, pd.DataFrame], configs: dict[str, HarnessConfig]
) -> CheckResult:
    """A run's row count must equal the number of items in its config's tasks.

    Items are matched to a task on benchmark == the task's benchmark (its name without
    _pinned, unless set) and benchmark_version == pinned yaml_version. Public runs and
    runs whose config_id doesn't resolve are skipped here; check_references reports
    the latter.
    """
    result = CheckResult("row count per run equals the item count")
    items_per_benchmark = (
        tables["items"]
        .drop_duplicates("item_id")
        .groupby(["benchmark", "benchmark_version"])
        .size()
    )
    expected = {
        config_id: sum(
            int(items_per_benchmark.get((t.benchmark, t.yaml_version), 0)) for t in config.tasks
        )
        for config_id, config in configs.items()
    }

    runs = (
        _own_rows(tables["long"])
        .groupby("run_id")
        .agg(rows=("item_id", "size"), config_id=("config_id", "first"))
    )
    runs["expected"] = runs["config_id"].map(expected)
    bad = runs[runs["expected"].notna() & (runs["rows"] != runs["expected"])]

    for r in bad.head(MAX_EXAMPLES).itertuples():
        result.failures.append(f"{r.Index}: {r.rows} rows, expected {int(r.expected)}")
    if len(bad) > MAX_EXAMPLES:
        result.failures.append(f"... and {len(bad) - MAX_EXAMPLES} more runs")
    return result


def check_references(
    tables: dict[str, pd.DataFrame],
    configs: dict[str, HarnessConfig],
    config_errors: list[str] = (),
) -> CheckResult:
    result = CheckResult("every item_id, model and config_id resolves in its table")
    long = tables["long"]

    unresolved = long[~long["item_id"].isin(tables["items"]["item_id"].dropna())]
    if len(unresolved):
        result.failures.append(
            f"item_id: {unresolved['item_id'].nunique(dropna=False)} values not in items, "
            f"e.g. {_examples(unresolved, ['item_id'])}"
        )

    known_models = pd.MultiIndex.from_frame(tables["models"][MODEL_KEY])
    unresolved = long[~pd.MultiIndex.from_frame(long[MODEL_KEY]).isin(known_models)]
    if len(unresolved):
        result.failures.append(
            f"model: {len(unresolved.drop_duplicates(MODEL_KEY))} (model_id, model_revision) "
            f"pairs not in models, e.g. {_examples(unresolved, MODEL_KEY)}"
        )

    own = _own_rows(long)
    unresolved = own[~own["config_id"].isin(list(configs))]
    if len(unresolved):
        result.failures.append(
            f"config_id: {unresolved['config_id'].nunique(dropna=False)} values match no "
            f"config file, e.g. {_examples(unresolved, ['config_id'])}"
        )
    for error in config_errors:
        result.failures.append(f"config file doesn't load: {error}")
    return result


def run_checks(
    tables: dict[str, pd.DataFrame],
    configs: dict[str, HarnessConfig],
    config_errors: list[str] = (),
) -> list[CheckResult]:
    return [
        check_unique_keys(tables["long"]),
        check_run_counts(tables, configs),
        check_references(tables, configs, config_errors),
    ]
