from __future__ import annotations
#python plot_tsv.py --tsv ./out/amine_split/primary_scan/primary_aliphatic_amines.tsv --output-dir ./out/amine_split/primary_scan/scan_summary/
# Total unique SMILES: 619
# Found in papers only: 18
#python plot_tsv.py --tsv ./out/amine_split/secondary_scan/secondary_aliphatic_amines.tsv --output-dir  ./out/amine_split/secondary_scan/scan_summary/
# Total unique SMILES: 5394
# Found in papers only: 43
#python plot_tsv.py --tsv ./out/amine_split/aromatic_scan/aromatic.tsv --output-dir  ./out/amine_split/aromatic_scan/scan_summary/
# Total unique SMILES: 7802
# Found in papers only: 45
#python plot_tsv.py --tsv ./out/amine_split/others_scan/others.tsv --output-dir  ./out/amine_split/others_scan/scan_summary/
# Total unique SMILES: 52519
# Found in papers only: 155

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class TsvSummary:
    tsv_path: Path
    total_smiles: int
    papers_only: int
    patents_only: int
    papers_and_patents: int
    not_found: int
    found_smiles: set[str]
    not_found_smiles: set[str]
    category_by_smiles: dict[str, str]


def discover_report_tsvs(input_dir: Path) -> list[Path]:
    report_tsvs: list[Path] = []
    for path in sorted(input_dir.rglob("*.tsv")):
        if path.parent == input_dir:
            continue
        if path.name.endswith("_summary.tsv"):
            continue
        report_tsvs.append(path)
    return report_tsvs


def summarize_report_tsv(tsv_path: Path) -> tuple[TsvSummary, pd.DataFrame]:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"InputSMILES", "MentionCount", "SourceType"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{tsv_path} is missing required columns: {sorted(missing)}")

    per_smiles: dict[str, dict[str, object]] = {}

    for row in df.to_dict(orient="records"):
        smiles = (row.get("InputSMILES") or "").strip()
        if not smiles:
            continue
        label = (row.get("Label") or "").strip()

        state = per_smiles.setdefault(
            smiles,
            {
                "label": label,
                "has_paper": False,
                "has_patent": False,
                "mention_count": 0,
            },
        )
        if label and not state["label"]:
            state["label"] = label

        try:
            mention_count = int((row.get("MentionCount") or "0").strip() or "0")
        except ValueError:
            mention_count = 0
        if mention_count > int(state["mention_count"]):
            state["mention_count"] = mention_count

        source_type = (row.get("SourceType") or "").strip().lower()
        if source_type == "paper":
            state["has_paper"] = True
        elif source_type == "patent":
            state["has_patent"] = True

    papers_only = 0
    patents_only = 0
    papers_and_patents = 0
    not_found = 0
    found_smiles: set[str] = set()
    not_found_smiles: set[str] = set()
    category_by_smiles: dict[str, str] = {}

    for smiles, state in per_smiles.items():
        if state["has_paper"] and state["has_patent"]:
            papers_and_patents += 1
            found_smiles.add(smiles)
            category_by_smiles[smiles] = "both"
        elif state["has_paper"]:
            papers_only += 1
            found_smiles.add(smiles)
            category_by_smiles[smiles] = "papers_only"
        elif state["has_patent"]:
            patents_only += 1
            found_smiles.add(smiles)
            category_by_smiles[smiles] = "patents_only"
        else:
            not_found += 1
            not_found_smiles.add(smiles)
            category_by_smiles[smiles] = "not_found"

    summary = TsvSummary(
        tsv_path=tsv_path,
        total_smiles=len(per_smiles),
        papers_only=papers_only,
        patents_only=patents_only,
        papers_and_patents=papers_and_patents,
        not_found=not_found,
        found_smiles=found_smiles,
        not_found_smiles=not_found_smiles,
        category_by_smiles=category_by_smiles,
    )
    return summary, df


def summary_to_lines(summary: TsvSummary) -> list[str]:
    found_any = summary.papers_only + summary.patents_only + summary.papers_and_patents
    return [
        f"TSV\t{summary.tsv_path}",
        f"TotalSMILES\t{summary.total_smiles}",
        f"FoundAny\t{found_any}",
        f"PapersOnly\t{summary.papers_only}",
        f"PatentsOnly\t{summary.patents_only}",
        f"PapersAndPatents\t{summary.papers_and_patents}",
        f"NotFound\t{summary.not_found}",
    ]


def write_summary_files(summary: TsvSummary, df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = summary.tsv_path.stem

    found_path = out_dir / f"{stem}_found.tsv"
    not_found_path = out_dir / f"{stem}_not_found.tsv"

    found_df = df[df["InputSMILES"].isin(summary.found_smiles)].copy()
    if not found_df.empty:
        found_df.insert(
            len(found_df.columns),
            "Category",
            found_df["InputSMILES"].map(summary.category_by_smiles).fillna("found"),
        )
        found_df = found_df.sort_values(
            by=["Category", "Label", "InputSMILES", "SourceType", "ReferenceID"],
            kind="stable",
        )
    found_df.to_csv(found_path, sep="\t", index=False)

    not_found_df = df[df["InputSMILES"].isin(summary.not_found_smiles)].copy()
    if not not_found_df.empty:
        not_found_df = not_found_df.drop_duplicates(subset=["InputSMILES"], keep="first").copy()
        not_found_df.insert(
            len(not_found_df.columns),
            "Category",
            not_found_df["InputSMILES"].map(summary.category_by_smiles).fillna("not_found"),
        )
        not_found_df = not_found_df.sort_values(by=["Label", "InputSMILES"], kind="stable")
    not_found_df.to_csv(not_found_path, sep="\t", index=False)
    return found_path, not_found_path


def plot_summary(summary: TsvSummary, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{summary.tsv_path.stem}_mentions_plot.png"

    labels = ["Papers", "Patents", "Both", "Not found"]
    values = [
        summary.papers_only,
        summary.patents_only,
        summary.papers_and_patents,
        summary.not_found,
    ]
    colors = ["#2563eb", "#f59e0b", "#7c3aed", "#9ca3af"]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title(f"Mention Summary: {summary.tsv_path.stem}", fontsize=16, weight="bold", pad=14)
    ax.set_ylabel("Unique SMILES")
    ax.set_xlabel("Category")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(values + [1]) + max(1, int(0.05 * max(values + [1]))))

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(value),
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

    found_any = summary.papers_only + summary.patents_only + summary.papers_and_patents
    ax.text(
        0.99,
        0.98,
        f"Total={summary.total_smiles} | Found={found_any} | Not found={summary.not_found}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def process_one_tsv(tsv_path: Path, output_dir: Path) -> None:
    summary, df = summarize_report_tsv(tsv_path)
    found_any = summary.papers_only + summary.patents_only + summary.papers_and_patents
    target_dir = output_dir / tsv_path.stem
    found_file, not_found_file = write_summary_files(summary, df, target_dir)
    plot_file = plot_summary(summary, target_dir)

    print(f"TSV: {tsv_path}")
    print(f"Total unique SMILES: {summary.total_smiles}")
    print(f"Found in papers only: {summary.papers_only}")
    print(f"Found in patents only: {summary.patents_only}")
    print(f"Found in both: {summary.papers_and_patents}")
    print(f"Found any: {found_any}")
    print(f"Not found: {summary.not_found}")
    print(f"Wrote found TSV: {found_file}")
    print(f"Wrote not-found TSV: {not_found_file}")
    print(f"Wrote plot: {plot_file}")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot mention-report TSV contents as papers/patents/both/not-found counts.",
    )
    parser.add_argument(
        "--input-dir",
        default="./out/amine_split",
        help="Directory to scan for report TSV files.",
    )
    parser.add_argument(
        "--tsv",
        default=None,
        help="Optional single TSV file to process instead of scanning a directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="./out/plots/scan_audit",
        help="Directory for generated summaries and plots.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.tsv:
        tsv_paths = [Path(args.tsv)]
    else:
        tsv_paths = discover_report_tsvs(Path(args.input_dir))

    if not tsv_paths:
        print(f"No report TSV files found in {Path(args.input_dir).resolve()}")
        return 1

    for tsv_path in tsv_paths:
        process_one_tsv(tsv_path, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
