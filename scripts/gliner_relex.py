"""Run zero-shot GLiNER-Relex with fixed labels and thresholds."""

import argparse
import csv
import json
import re
from pathlib import Path

MODEL = "knowledgator/gliner-relex-large-v0.5"
REVISION = "9c4171ae1e690fc29b87f33579e50bcd65faf2cc"
ENTITY_LABELS = {
    "variable": "A psychological variable, construct, behavior, intervention, condition, or measured attribute studied in the abstract."
}
RELATION_LABELS = {
    "predicts or affects": "directional",
    "is associated with": "correlational",
    "has component or subtype": "hierarchy",
    "moderates": "moderation",
}
SETTINGS = dict(
    threshold=0.3,
    relation_threshold=0.5,
    adjacency_threshold=0.5,
    flat_ner=False,
    multi_label=False,
    batch_size=1,
    return_relations=True,
)


def graph_row(article, entities, relations):
    """Serialize native entity and binary-relation predictions for graph_eval."""
    nodes = {}

    def node(value):
        label = re.sub(r"\s+", " ", value["text"].strip())
        label = label.replace("<->", "↔").replace("->", "→")
        label = label.replace(";", " ").replace(",", " ")
        label = label.replace("[", "(").replace("]", ")")
        if label.count("(") != label.count(")"):
            label = label.replace("(", "").replace(")", "")
        label = re.sub(r"\s+", " ", label).strip()
        if not label:
            raise ValueError("Empty node label")
        return nodes.setdefault(label.casefold(), label)

    for entity in entities:
        node(entity)
    edges = {kind: set() for kind in RELATION_LABELS.values()}
    for relation in relations:
        kind = RELATION_LABELS[relation["relation"]]
        head, tail = node(relation["head"]), node(relation["tail"])
        if head == tail:
            continue
        if kind == "correlational":
            head, tail = sorted((head, tail), key=str.casefold)
        edges[kind].add((head, tail))

    row = {
        "article_id": f"json_{article['id']}",
        "title": article.get("data", {}).get("TI", ""),
        "abstract": article.get("data", {}).get("AB", ""),
        "status": "completed",
        "variables": "; ".join(nodes.values()),
        "step2_final_vars": "; ".join(nodes.values()),
    }
    for kind in ("hierarchy", "directional", "correlational"):
        symbol = "<->" if kind == "correlational" else "->"
        row[f"step5_1_{kind}"] = "; ".join(
            f"{a} {symbol} {b}" for a, b in sorted(edges[kind])
        )
    # Preserve the model's native binary moderation target; do not invent a
    # second endpoint to turn it into a three-node moderated relationship.
    row["step5_1_moderation"] = "; ".join(
        f"{m} moderates ({v})" for m, v in sorted(edges["moderation"])
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/EmpiriGraph-Psy_gold_annotation.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/output/gliner_relex.csv")
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    from gliner import GLiNER  # optional dependency: gliner==0.2.29

    articles = json.loads(args.input.read_text(encoding="utf-8"))
    model = GLiNER.from_pretrained(
        MODEL, revision=REVISION, map_location=args.device, strict=True
    )
    model.eval()
    rows = []
    for article in articles:
        abstract = article.get("data", {}).get("AB", "")
        if not abstract:
            continue
        entities, relations = model.inference(
            texts=[abstract],
            labels=ENTITY_LABELS,
            relations=list(RELATION_LABELS),
            **SETTINGS,
        )
        rows.append(graph_row(article, entities[0], relations[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
