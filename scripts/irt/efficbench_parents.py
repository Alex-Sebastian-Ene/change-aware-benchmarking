"""Add parent models missing from the efficbench MMLU matrix, from public leaderboard details.

Downloads each model's latest full MMLU run from open-llm-leaderboard-old into
data/raw/leaderboard_old_mmlu/ and writes the new rows to
data/processed/efficbench_mmlu_parents/. Models already in the matrix are the check: their
downloaded answers must match it, and every run must use the same questions in the same order.

Run: uv run --group matrix python -I scripts/irt/efficbench_parents.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/leaderboard_old_mmlu"
MATRIX = ROOT / "data/processed/efficbench_mmlu"
OUT = ROOT / "data/processed/efficbench_mmlu_parents"
N_SUBJECTS = 57
MIN_RUN_BYTES = 10e6  # limited test runs are about 1 MB; full MMLU runs are over 20 MB
MIN_AGREEMENT = 0.999

MODELS = [
    "01-ai/Yi-34B",
    "HuggingFaceH4/zephyr-7b-beta",
    "Intel/neural-chat-7b-v3-1",
    "Intel/neural-chat-7b-v3-3",
    "Mihaiii/Pallas-0.5",
    "Mihaiii/Pallas-0.5-LASER-0.3",
    "Mihaiii/Pallas-0.5-LASER-0.5",
    "Open-Orca/OpenOrca-Platypus2-13B",
    "OpenPipe/mistral-ft-optimized-1218",
    "Q-bert/MetaMath-Cybertron",
    "alignment-handbook/zephyr-7b-sft-full",
    "berkeley-nest/Starling-LM-7B-alpha",
    "codellama/CodeLlama-7b-hf",
    "fblgit/una-cybertron-7b-v2-bf16",
    "lmsys/vicuna-13b-v1.5",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-math/MetaMath-Mistral-7B",
    "microsoft/Orca-2-13b",
    "migtissera/Tess-34B-v1.4",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "mlabonne/NeuralHermes-2.5-Mistral-7B",
    "mlabonne/NeuralMarcoro14-7B",
    "one-man-army/una-neural-chat-v3-3-P1-OMA",
    "openchat/openchat-3.5-1210",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "tiiuae/falcon-7b",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "upstage/SOLAR-10.7B-v1.0",
]


def pick_run(files):
    """Latest run with every MMLU subject at full size. `files` is [(path, bytes)]."""
    runs = {}
    for path, size in files:
        if "hendrycksTest" in path and path.endswith(".parquet"):
            runs.setdefault(path.split("/")[0], []).append((path, size))
    full = [
        run
        for run, parts in runs.items()
        if len(parts) == N_SUBJECTS and sum(size for _, size in parts) >= MIN_RUN_BYTES
    ]
    return [path for path, _ in runs[max(full)]] if full else []


def download(api, model):
    repo = "open-llm-leaderboard-old/details_" + model.replace("/", "__")
    info = api.dataset_info(repo, token=False, files_metadata=True)
    paths = pick_run([(s.rfilename, s.size) for s in info.siblings])
    if not paths:
        raise ValueError(f"{model} has no full MMLU run")
    folder = RAW / model.replace("/", "__")
    for path in paths:
        if not (folder / path).exists():
            hf_hub_download(repo, path, repo_type="dataset", local_dir=folder, token=False)
    return folder / paths[0].split("/")[0]


def read_run(folder, subjects):
    """Answers (0/1) and question text in matrix order, one subject file at a time."""
    answers, questions = [], []
    files = {f.name.split("hendrycksTest-")[1].split("|")[0]: f for f in folder.iterdir()}
    for subject in subjects:
        file = pq.ParquetFile(files[subject])
        columns = [c for c in ("acc", "metrics", "example") if c in file.schema_arrow.names]
        for row in file.read(columns=columns).to_pylist():
            answers.append(row["acc"] if "acc" in row else row["metrics"]["acc"])
            questions.append(row["example"])
    return np.array(answers, dtype=float), questions


def main():
    Y = np.load(MATRIX / "responses.npy")
    names = pd.read_csv(MATRIX / "models.csv").model_id.tolist()
    subjects = list(dict.fromkeys(pd.read_csv(MATRIX / "items.csv").subject))
    api = HfApi(token=False)
    new, new_names, reference = [], [], None
    for model in MODELS:
        answers, questions = read_run(download(api, model), subjects)
        assert len(answers) == Y.shape[1], f"{model}: {len(answers)} answers"
        assert set(np.unique(answers)) <= {0.0, 1.0}, f"{model}: answers are not 0/1"
        reference = reference or questions
        assert questions == reference, f"{model}: different questions or order"
        if model in names:
            agreement = (answers == Y[names.index(model)]).mean()
            assert agreement >= MIN_AGREEMENT, f"{model}: agrees on only {agreement:.4f}"
            print(f"checked {model}: {agreement:.4f} agreement", flush=True)
        else:
            new.append(answers)
            new_names.append(model)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "responses.npy", np.array(new, dtype=np.int8))
    pd.DataFrame({"model_id": new_names}).to_csv(OUT / "models.csv", index=False)
    print(f"Saved {len(new)} new models to {OUT}")


if __name__ == "__main__":
    main()
