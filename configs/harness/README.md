# Harness configs

One YAML file per configuration. The file's sha256 is the `config_id`.

Files are never edited in place. The filename must end in the first 12 characters of
the file's hash (`<name>-<hash12>.yaml`) and `load_config` refuses a file whose name
doesn't match, so any edit forces a new file and a new `config_id`.

To create one: copy `template.yaml.example`, fill it in, then get its name with

    uv run python -c "from pathlib import Path; from cabench.harness.pins import expected_filename; print(expected_filename(Path('configs/harness/draft.yaml')))"

and rename the file to that.

Fill in every `<...>` placeholder before renaming: `load_config` refuses a config
unless `lm_eval_sha` and each `dataset_sha` are full 40-char commits, `lockfile_sha256`
and `apptainer_digest` are full sha256s, and `vllm_version` is an exact version.
Quote the version (`"0.7"`): unquoted, YAML reads `0.7` as a number.

Each task's `benchmark` (the name in the long and items tables, and in the `item_id`
hash, so it must match the public runs) defaults to the task name without its
`_pinned` suffix: `leaderboard_mmlu_pro_pinned` -> `leaderboard_mmlu_pro`. Set
`benchmark:` on the task only if that rule gives the wrong name.

The matrix scripts (`scripts/matrix/utils/config.py`) use these same rules and IDs.
