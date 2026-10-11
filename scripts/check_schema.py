"""Schema check for a new batch of response tables (acceptance test 3).

    uv run python scripts/check_schema.py [tables_dir] [--configs configs/harness]

tables_dir defaults to data/processed. Exit code 0 = all checks pass, 1 = a check
failed, 2 = the tables or configs couldn't be read. Layout: cabench/schema_check.py.
"""

import argparse
import sys
from pathlib import Path

from cabench.schema_check import load_configs, read_tables, run_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Schema check for the response tables.")
    parser.add_argument("tables_dir", type=Path, nargs="?", default=Path("data/processed"))
    parser.add_argument("--configs", type=Path, default=Path("configs/harness"))
    args = parser.parse_args(argv)

    if not args.configs.is_dir():
        print(f"ERROR  configs directory {args.configs} not found", file=sys.stderr)
        return 2
    try:
        tables = read_tables(args.tables_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR  {e}", file=sys.stderr)
        return 2
    configs, config_errors = load_configs(args.configs)

    responses = tables["responses"]
    print(
        f"{args.tables_dir}: {len(responses)} rows, {responses['run_id'].nunique()} runs, "
        f"{len(tables['models'])} models, {len(tables['items'])} items, {len(configs)} configs"
    )
    results = run_checks(tables, configs, config_errors)
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.name}")
        for line in r.failures:
            print(f"      {line}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
