"""CLI entry point for CPA pathway survey analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# Common string values that should be treated as missing across survey columns.
MISSING_TOKENS = {"", "na", "n/a", "null", "none"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the end-to-end pipeline."""
    parser = argparse.ArgumentParser(description="Run CPA pathway survey analysis")
    parser.add_argument("--config", required=True, help="Path to columns.yml")
    parser.add_argument("--input", required=True, help="Path to source CSV file")
    parser.add_argument("--out", required=True, help="Output directory path")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config that defines exact/regex column matching rules."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(base_out: Path) -> None:
    """Create output folders for tables and figures if they do not already exist."""
    (base_out / "tables").mkdir(parents=True, exist_ok=True)
    (base_out / "figures").mkdir(parents=True, exist_ok=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize missing values, strip whitespace, and drop Qualtrics metadata rows."""
    cleaned = df.copy()

    # Normalize strings first so missing-token replacement is consistent.
    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].astype(str).str.strip()

    # Replace known missing markers with NaN using a case-insensitive check.
    def normalize_missing(value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in MISSING_TOKENS:
            return np.nan
        return value

    cleaned = cleaned.applymap(normalize_missing)

    # Qualtrics exports include two metadata rows directly under the header.
    if "ResponseId" in cleaned.columns:
        response_id = cleaned["ResponseId"].fillna("").astype(str)
        metadata_mask = response_id.eq("Response ID") | response_id.str.contains("ImportId", na=False)
        cleaned = cleaned.loc[~metadata_mask].copy()

    return cleaned


def resolve_column(df: pd.DataFrame, pattern: str) -> str | None:
    """Resolve a column by exact match first, then with regex fallback."""
    # Exact matching allows users to pin a field if headers are stable.
    for column in df.columns:
        if column == pattern:
            return column

    # Regex matching provides flexibility for changing survey question text.
    regex = re.compile(pattern, flags=0)
    for column in df.columns:
        if regex.search(column):
            return column
    return None


def resolve_columns(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve required and optional columns from configuration rules."""
    resolved: dict[str, Any] = {}
    resolved["stance_q6"] = resolve_column(df, config["stance_q6"])
    resolved["stance_reason_q50"] = resolve_column(df, config["stance_reason_q50"])
    resolved["awareness"] = resolve_column(df, config["awareness"])
    resolved["cpa_likelihood"] = resolve_column(df, config["cpa_likelihood"])
    resolved["grad_likelihood"] = resolve_column(df, config["grad_likelihood"])

    # Optional segment columns are resolved independently and de-duplicated.
    segment_matches = []
    for segment_pattern in config.get("segments", []):
        matched = resolve_column(df, segment_pattern)
        if matched and matched not in segment_matches:
            segment_matches.append(matched)
    resolved["segments"] = segment_matches
    return resolved


def normalize_stance_label(value: Any) -> str | None:
    """Map raw Q6 responses into favorable/neutral/unfavorable categories."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    # Numeric or numeric-text Likert mappings: 1-2 unfavorable, 3 neutral, 4-5 favorable.
    numeric_match = re.search(r"\b([1-5])(\.0+)?\b", text)
    if numeric_match and text.lower() not in {"q6"}:
        score = int(float(numeric_match.group(1)))
        if score >= 4:
            return "favorable"
        if score == 3:
            return "neutral"
        return "unfavorable"

    lower = text.lower()
    if any(token in lower for token in ["very positive", "somewhat positive", "positive", "like", "favor"]):
        return "favorable"
    if any(token in lower for token in ["very negative", "somewhat negative", "negative", "dislike", "oppose"]):
        return "unfavorable"
    if "neutral" in lower or "neither" in lower:
        return "neutral"

    # Unknown labels are left as missing to avoid forced misclassification.
    return None


def compute_stance_distribution(df: pd.DataFrame, stance_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute count and percent distribution across stance groups."""
    work = df.copy()
    work["stance_group"] = work[stance_col].apply(normalize_stance_label)
    valid = work["stance_group"].dropna()
    total_n = int(valid.shape[0])

    counts = valid.value_counts().reindex(["favorable", "neutral", "unfavorable"], fill_value=0)
    distribution = pd.DataFrame(
        {
            "stance_group": counts.index,
            "count": counts.values,
            "percent": (counts.values / total_n * 100.0) if total_n else np.zeros(len(counts)),
        }
    )

    summary = {
        "total_n": total_n,
        "favorable_pct": float(distribution.loc[distribution["stance_group"] == "favorable", "percent"].iloc[0]),
        "neutral_pct": float(distribution.loc[distribution["stance_group"] == "neutral", "percent"].iloc[0]),
        "unfavorable_pct": float(distribution.loc[distribution["stance_group"] == "unfavorable", "percent"].iloc[0]),
    }
    return distribution, summary


def plot_stance_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    """Create and save a bar chart for stance distribution percentages."""
    plt.figure(figsize=(8, 5))
    plt.bar(distribution["stance_group"], distribution["percent"], color=["#2E8B57", "#808080", "#B22222"])
    plt.ylabel("Percent")
    plt.title("Stance Distribution on Alternative CPA Pathway")
    plt.ylim(0, max(5, float(distribution["percent"].max() + 5)))
    for idx, row in distribution.iterrows():
        plt.text(idx, row["percent"] + 0.5, f"{row['percent']:.1f}%\nN={int(row['count'])}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def clean_text(text: Any) -> str:
    """Normalize open-text responses for term and topic extraction."""
    if pd.isna(text):
        return ""
    lowered = str(text).lower()
    no_punct = re.sub(r"[^a-z0-9\s]", " ", lowered)
    collapsed = re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


def top_terms_by_stance(df: pd.DataFrame, reason_col: str) -> pd.DataFrame:
    """Compute top TF-IDF terms (top 20) for each stance group."""
    if reason_col not in df.columns or "stance_group" not in df.columns:
        return pd.DataFrame(columns=["stance_group", "term", "weight"])

    rows: list[dict[str, Any]] = []
    for group in ["favorable", "neutral", "unfavorable"]:
        texts = df.loc[df["stance_group"] == group, reason_col].dropna().astype(str).map(clean_text)
        texts = texts[texts.str.len() > 0]
        if texts.empty:
            continue

        # Group-level TF-IDF uses responses within each stance segment.
        vectorizer = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2), max_features=1500)
        matrix = vectorizer.fit_transform(texts)
        term_weights = np.asarray(matrix.mean(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        top_indices = np.argsort(term_weights)[::-1][:20]

        for idx in top_indices:
            rows.append({"stance_group": group, "term": terms[idx], "weight": float(term_weights[idx])})

    return pd.DataFrame(rows)


def build_topics(df: pd.DataFrame, reason_col: str, n_topics: int = 8, n_terms: int = 10) -> pd.DataFrame:
    """Run lightweight NMF topic modeling on open-text reason responses."""
    if reason_col not in df.columns:
        return pd.DataFrame(columns=["topic_id", "top_terms"])

    texts = df[reason_col].dropna().astype(str).map(clean_text)
    texts = texts[texts.str.len() > 0]
    if texts.shape[0] < 10:
        return pd.DataFrame(columns=["topic_id", "top_terms"])

    # Use a broad TF-IDF representation for simple, interpretable topics.
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.9, max_features=2000)
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return pd.DataFrame(columns=["topic_id", "top_terms"])

    # Keep topic count bounded by available signal to prevent model errors.
    topic_count = max(2, min(n_topics, matrix.shape[0] - 1, matrix.shape[1] - 1))
    model = NMF(n_components=topic_count, random_state=42, init="nndsvda", max_iter=400)
    model.fit(matrix)
    features = np.array(vectorizer.get_feature_names_out())

    records = []
    for topic_id, component in enumerate(model.components_, start=1):
        top = features[np.argsort(component)[::-1][:n_terms]]
        records.append({"topic_id": topic_id, "top_terms": ", ".join(top)})
    return pd.DataFrame(records)


def compute_favorable_by_segment(df: pd.DataFrame, segment_columns: list[str]) -> pd.DataFrame:
    """Compute favorable share by subgroup for each available segment column."""
    rows: list[dict[str, Any]] = []

    for segment_col in segment_columns:
        if segment_col not in df.columns:
            continue

        # Keep non-missing segment values to avoid artificial unknown groups.
        segment_df = df[[segment_col, "stance_group"]].dropna(subset=[segment_col]).copy()
        if segment_df.empty:
            continue

        grouped = segment_df.groupby(segment_col, dropna=True)
        for group_name, group_values in grouped:
            valid = group_values["stance_group"].dropna()
            n = int(valid.shape[0])
            if n == 0:
                continue
            favorable_rate = float((valid == "favorable").mean() * 100.0)
            rows.append(
                {
                    "segment_column": segment_col,
                    "segment_group": str(group_name),
                    "n": n,
                    "favorable_percent": favorable_rate,
                }
            )

    return pd.DataFrame(rows)


def plot_best_segment(segment_df: pd.DataFrame, output_path: Path) -> str | None:
    """Plot favorable percentage for the segment column with the largest covered sample."""
    if segment_df.empty:
        return None

    # Pick segment with highest total N across its groups.
    coverage = segment_df.groupby("segment_column")["n"].sum().sort_values(ascending=False)
    best_segment = str(coverage.index[0])
    best_df = segment_df.loc[segment_df["segment_column"] == best_segment].sort_values("favorable_percent", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(best_df["segment_group"].astype(str), best_df["favorable_percent"], color="#1f77b4")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Favorable %")
    plt.title(f"Favorable Rate by Segment: {best_segment}")
    for idx, row in enumerate(best_df.itertuples(index=False)):
        plt.text(idx, row.favorable_percent + 0.5, f"N={row.n}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return best_segment


def format_percent(value: float) -> str:
    """Format percentage with one decimal for markdown reporting."""
    return f"{value:.1f}%"


def build_report(
    summary: dict[str, Any],
    distribution: pd.DataFrame,
    top_terms: pd.DataFrame,
    topics: pd.DataFrame,
    segment_df: pd.DataFrame,
    resolved_columns: dict[str, Any],
    out_path: Path,
) -> None:
    """Generate a markdown report with headline metrics, themes, segments, and caveats."""
    # Pull concise top terms for favorable/unfavorable audiences.
    favorable_terms = top_terms.loc[top_terms["stance_group"] == "favorable", "term"].head(5).tolist()
    unfavorable_terms = top_terms.loc[top_terms["stance_group"] == "unfavorable", "term"].head(5).tolist()

    # Add topic keywords to provide extra context for themes.
    topic_keywords = []
    for _, row in topics.head(3).iterrows():
        topic_keywords.append(row["top_terms"].split(", ")[0])

    # Highlight biggest segment differences using groups with adequate sample sizes.
    segment_highlights = []
    if not segment_df.empty:
        filtered = segment_df.loc[segment_df["n"] >= 10].sort_values("favorable_percent", ascending=False)
        top_rows = filtered.head(3)
        for _, row in top_rows.iterrows():
            segment_highlights.append(
                f"{row['segment_column']} = {row['segment_group']}: {row['favorable_percent']:.1f}% favorable (N={int(row['n'])})"
            )

    lines = [
        "# Alternative CPA Pathway Sentiment Report",
        "",
        "## Headline",
        (
            f"Among respondents with a codable stance (N={summary['total_n']}), "
            f"{format_percent(summary['favorable_pct'])} are favorable, "
            f"{format_percent(summary['unfavorable_pct'])} are unfavorable, and "
            f"{format_percent(summary['neutral_pct'])} are neutral."
        ),
        "",
        "## Stance distribution",
    ]

    for _, row in distribution.iterrows():
        lines.append(f"- {row['stance_group'].title()}: {row['count']} respondents ({row['percent']:.1f}%)")

    lines.extend([
        "",
        "## Top themes",
        f"- Favorable themes: {', '.join(favorable_terms) if favorable_terms else 'Not enough text data.'}",
        f"- Unfavorable themes: {', '.join(unfavorable_terms) if unfavorable_terms else 'Not enough text data.'}",
        f"- Topic keywords: {', '.join(topic_keywords) if topic_keywords else 'Not enough text data for topics.'}",
        "",
        "## Who likes it more",
    ])

    if segment_highlights:
        for item in segment_highlights:
            lines.append(f"- {item}")
    else:
        lines.append("- Segment data was missing or subgroup sizes were too small for reliable comparison.")

    lines.extend([
        "",
        "## Caveats",
        "- Missing data and blank responses reduce analyzable sample sizes across outputs.",
        "- Small subgroup sizes can make segment-level percentages unstable.",
        "- Stance mapping assumptions convert Q6 responses into favorable/neutral/unfavorable buckets.",
        f"- Column matching relied on regex configuration and resolved as: `{json.dumps(resolved_columns, default=str)}`.",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(config_path: Path, input_path: Path, output_dir: Path) -> None:
    """Execute the full analysis pipeline and write all requested deliverables."""
    ensure_output_dirs(output_dir)

    # Load config and data before applying cleaning and flexible column resolution.
    config = load_config(config_path)
    raw_df = pd.read_csv(input_path, dtype=str)
    df = clean_dataframe(raw_df)
    resolved = resolve_columns(df, config)

    # Save column resolution details for transparency/debugging.
    (output_dir / "column_matches.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    stance_col = resolved.get("stance_q6")
    if not stance_col:
        raise ValueError("Could not resolve required stance column (stance_q6).")

    distribution, summary = compute_stance_distribution(df, stance_col)
    df["stance_group"] = df[stance_col].apply(normalize_stance_label)

    # Write required stance outputs even if optional inputs are unavailable.
    distribution.to_csv(output_dir / "tables" / "stance_distribution.csv", index=False)
    plot_stance_distribution(distribution, output_dir / "figures" / "stance_distribution.png")

    # Build term and topic outputs, gracefully falling back to empty tables.
    reason_col = resolved.get("stance_reason_q50")
    top_terms = top_terms_by_stance(df, reason_col) if reason_col else pd.DataFrame(columns=["stance_group", "term", "weight"])
    top_terms.to_csv(output_dir / "tables" / "top_terms_by_stance.csv", index=False)

    topics = build_topics(df, reason_col) if reason_col else pd.DataFrame(columns=["topic_id", "top_terms"])
    topics.to_csv(output_dir / "tables" / "topics.csv", index=False)

    # Compute segmentation outputs when optional segment columns are available.
    segment_df = compute_favorable_by_segment(df, resolved.get("segments", []))
    segment_df.to_csv(output_dir / "tables" / "stance_by_segment.csv", index=False)
    if not segment_df.empty:
        plot_best_segment(segment_df, output_dir / "figures" / "favorable_by_segment.png")
    else:
        # Ensure the expected file exists even when no segment chart can be generated.
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No segment data available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "favorable_by_segment.png", dpi=150)
        plt.close()

    build_report(
        summary=summary,
        distribution=distribution,
        top_terms=top_terms,
        topics=topics,
        segment_df=segment_df,
        resolved_columns=resolved,
        out_path=output_dir / "report.md",
    )


def main() -> None:
    """Main CLI entrypoint for running the analysis pipeline."""
    args = parse_args()
    run_pipeline(Path(args.config), Path(args.input), Path(args.out))


if __name__ == "__main__":
    main()
