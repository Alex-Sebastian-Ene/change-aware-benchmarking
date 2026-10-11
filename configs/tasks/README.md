# Task override YAMLs

lm-eval loads these via `TaskManager(include_path="configs/tasks")`.

One file per pinned task: a copy of the upstream task YAML at the pinned lm-eval
commit, renamed (`task: <name>_pinned`) and with the dataset revision pinned:

```yaml
dataset_kwargs:
  revision: <40-char dataset commit sha>
```

Copy rather than `include:` the upstream file, because `include:` paths are resolved
relative to the file and would point into a machine-specific install.
