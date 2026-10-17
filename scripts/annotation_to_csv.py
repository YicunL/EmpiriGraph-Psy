"""Convert Label Studio graph annotations to one CSV row per abstract."""

import argparse
import csv
import json
from pathlib import Path


COLUMNS = (
    "source_file", "task_id", "ORN", "TI", "YR", "AU", "JN", "AB", "MD",
    "Methodology_Category", "Period", "annotator_id", "n_variables",
    "variables", "n_hierarchy", "hierarchy", "n_directional", "directional",
    "n_correlational", "correlational", "n_moderation", "moderation",
)


def convert(item: dict, source_file: str) -> dict:
    data = item.get("data", {})
    row = {key: data.get(key, "") for key in COLUMNS[2:11]}
    row.update(source_file=source_file, task_id=item.get("id", ""))
    annotation = (item.get("annotations") or [{}])[0]
    result = annotation.get("result", [])
    row["annotator_id"] = annotation.get("completed_by", "")

    variables = {
        entry["id"]: entry.get("value", {}).get("text", "").strip()
        for entry in result
        if entry.get("type") == "labels"
        and "Variable" in entry.get("value", {}).get("labels", [])
    }
    for entry in result:
        if entry.get("type") == "textarea" and entry.get("from_name") == "var_name":
            if entry.get("id") in variables:
                name = entry.get("value", {}).get("text", "")
                if isinstance(name, list):
                    name = name[0] if name else ""
                if name.strip():
                    variables[entry["id"]] = name.strip()

    edges = {kind: [] for kind in ("hierarchy", "directional", "correlational", "moderation")}
    moderation = {}
    for entry in result:
        if entry.get("type") != "relation":
            continue
        source = variables.get(entry.get("from_id"))
        target = variables.get(entry.get("to_id"))
        if not source or not target:
            continue
        for label in entry.get("labels", []):
            kind, _, state = label.partition("__")
            if kind == "hierarchy":
                edges["hierarchy"].append(f"{source} -> {target}")
            elif kind.startswith("direction"):
                edges["directional"].append(f"{source} -> {target} [{state}]")
            elif kind.startswith("correlation"):
                edges["correlational"].append(f"{source} <-> {target} [{state}]")
            elif kind.startswith("moderation"):
                moderation.setdefault((source, state), []).append(target)
    edges["moderation"] = [
        f"{source} moderates ({', '.join(targets)}) [{state}]"
        for (source, state), targets in moderation.items()
    ]
    row["n_variables"] = len(variables)
    row["variables"] = "; ".join(sorted(variables.values()))
    for kind, values in edges.items():
        row[f"n_{kind}"] = len(values)
        row[kind] = "; ".join(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--annotator", help="Annotator label written to annotator_id")
    args = parser.parse_args()
    rows = [convert(item, args.input.name) for item in json.loads(args.input.read_text(encoding="utf-8"))]
    if args.annotator:
        for row in rows:
            row["annotator_id"] = args.annotator
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
