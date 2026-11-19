"""Real parent and fine-tuned child pairs inside the efficbench MMLU matrix.

efficbench_lineage.csv lists each link and its evidence (Hub base_model metadata, or the
child's name or card naming its parent). `fine-tuned` and `version step` rows are tested;
`related` rows only keep relatives in the same fold. Merges, re-uploads and ambiguous names
are left out, and no pair was picked by how it scored. Parents added by
efficbench_parents.py are used when present, and pairs missing either model are skipped.

Run: uv run --group irt python scripts/irt/efficbench_pairs.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from efficbench import ROOT, load
from run_experiment import FAMILY_DIST, KS, SEEDS
from run_pairs import CHECK_K, FOCUS, N_DIMS, make_folds, run
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cabench.pairs import drop_near_duplicates
from cabench.splits import behaviour_families

LINEAGE = Path(__file__).with_name("efficbench_lineage.csv")
PARENTS = ROOT / "data/processed/efficbench_mmlu_parents"


def lineage_groups(names, families, links):
    """Group id per model. Linked models, shared bases and behavioural families share one."""
    family_nodes = [f"family {f}" for f in families]
    edges = pd.concat(
        [
            pd.DataFrame({"a": names, "b": family_nodes}),
            links[["child", "parent"]].set_axis(["a", "b"], axis=1),
        ]
    )
    codes, nodes = pd.factorize(pd.concat([edges.a, edges.b]))
    a, b = np.split(codes, 2)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes), len(nodes)))
    labels = connected_components(graph, directed=False)[1]
    return dict(zip(nodes, labels, strict=True))


def load_with_parents():
    """Responses, model names, items and output folder name, plus downloaded parents."""
    Y, models, items = load()
    names, out = models.model_id.to_numpy(), "efficbench_mmlu"
    if (PARENTS / "responses.npy").exists():
        Y = np.vstack([Y, np.load(PARENTS / "responses.npy")])
        names = np.r_[names, pd.read_csv(PARENTS / "models.csv").model_id.to_numpy()]
        out = "efficbench_mmlu_extended"
    return Y.astype(np.float32), names, items, out


def main():
    Y, names, items, out = load_with_parents()
    links = pd.read_csv(LINEAGE)
    group = lineage_groups(names, behaviour_families(Y, FAMILY_DIST), links)

    tested = links[(links["use"] != "related") & links.child.isin(names)]
    tested = tested[tested.parent.isin(names)]
    pairs = pd.DataFrame(
        {
            "child": tested.child.values,
            "parent": tested.parent.values,
            "root": [f"group {group[c]}" for c in tested.child],
            "kind": np.where(tested["use"] == "fine-tuned", FOCUS, "version step"),
            "evidence": tested.evidence.values,
        }
    )
    pairs = drop_near_duplicates(pairs, Y, names)
    roots = np.array([f"group {group[n]}" for n in names])
    print(f"{len(pairs)} pairs in {pairs['root'].nunique()} separate lineage groups", flush=True)
    summary, _, checks = run(
        Y,
        names,
        pairs,
        roots,
        pd.factorize(items.subject)[0],
        ROOT / "results/irt/pairs" / out,
        N_DIMS,
        KS,
        SEEDS,
        make_folds(pairs),
    )
    shown = summary[(summary.kind == FOCUS) & (summary.k == CHECK_K)]
    print(shown[["method", "abs_error", "right_direction", "coverage_80", "coverage_95"]])
    print(checks)


if __name__ == "__main__":
    main()
