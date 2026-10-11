# Harness configs

One YAML file per configuration. The file's sha256 is the `config_id`.

Files are never edited in place. The filename must end in the first 12 characters of
the file's hash (`<name>-<hash12>.yaml`) and `load_config` refuses a file whose name
doesn't match, so any edit forces a new file and a new `config_id`.

To create one: copy `template.yaml.example`, fill it in, then get its name with

    uv run python -c "from pathlib import Path; from cabench.harness.pins import expected_filename; print(expected_filename(Path('configs/harness/draft.yaml')))"

and rename the file to that.
