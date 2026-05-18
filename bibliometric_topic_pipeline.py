"""
bibliometric_topic_pipeline.py

Purpose:
    A compact code sample for processing publication metadata, extracting
    technology-related keywords, clustering abstracts, and identifying
    emerging research topics over time.

Use case:
    This mirrors a simplified version of a bibliometric data workflow relevant
    to emerging technology analysis, publication mapping, and science-of-science
    research.

Inputs:
    CSV file with columns:
        - paper_id
        - title
        - abstract
        - year
        - authors
        - affiliations

Outputs:
    - cleaned_publications.csv
    - clustered_publications.csv
    - emerging_topics.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


@dataclass
class PipelineConfig:
    input_csv: Path
    output_dir: Path
    min_year: int = 2015
    max_features: int = 5000
    n_clusters: int = 12
    top_keywords_per_cluster: int = 12


def normalize_text(text: str) -> str:
    """Clean and normalize publication title/abstract text."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean_publications(input_csv: Path, min_year: int) -> pd.DataFrame:
    """Load publication metadata and perform basic cleaning."""
    df = pd.read_csv(input_csv)

    required_cols = {"paper_id", "title", "abstract", "year"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["paper_id", "title", "year"])
    df = df[df["year"] >= min_year]

    df["title_clean"] = df["title"].apply(normalize_text)
    df["abstract_clean"] = df["abstract"].apply(normalize_text)
    df["text"] = (df["title_clean"] + " " + df["abstract_clean"]).str.strip()

    df = df[df["text"].str.len() > 50]
    df = df.drop_duplicates(subset=["paper_id"])

    return df


def vectorize_text(
    texts: Iterable[str],
    max_features: int,
) -> tuple[TfidfVectorizer, np.ndarray]:
    """Convert publication text into TF-IDF vectors."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def cluster_publications(
    matrix,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Cluster publication vectors and return cluster labels."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = model.fit_predict(matrix)

    score = silhouette_score(matrix, labels) if n_clusters > 1 else np.nan
    return labels, score


def extract_cluster_keywords(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    top_n: int,
) -> pd.DataFrame:
    """Extract representative keywords for each cluster."""
    terms = np.array(vectorizer.get_feature_names_out())
    rows = []

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_idx = np.where(df["cluster"].values == cluster_id)[0]
        cluster_matrix = matrix[cluster_idx]

        mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_indices = mean_tfidf.argsort()[::-1][:top_n]
        keywords = terms[top_indices].tolist()

        rows.append(
            {
                "cluster": cluster_id,
                "n_papers": len(cluster_idx),
                "top_keywords": "; ".join(keywords),
            }
        )

    return pd.DataFrame(rows)


def detect_emerging_topics(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    recent_window: int = 3,
) -> pd.DataFrame:
    """
    Identify clusters with increasing recent publication activity.

    Growth score:
        recent share of papers - historical share of papers
    """
    max_year = int(df["year"].max())
    recent_start = max_year - recent_window + 1

    total_recent = max((df["year"] >= recent_start).sum(), 1)
    total_historical = max((df["year"] < recent_start).sum(), 1)

    rows = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]

        recent_count = ((cluster_df["year"] >= recent_start)).sum()
        historical_count = ((cluster_df["year"] < recent_start)).sum()

        recent_share = recent_count / total_recent
        historical_share = historical_count / total_historical
        growth_score = recent_share - historical_share

        rows.append(
            {
                "cluster": cluster_id,
                "recent_count": int(recent_count),
                "historical_count": int(historical_count),
                "recent_share": round(recent_share, 4),
                "historical_share": round(historical_share, 4),
                "growth_score": round(growth_score, 4),
            }
        )

    emerging = pd.DataFrame(rows)
    emerging = emerging.merge(cluster_summary, on="cluster", how="left")
    emerging = emerging.sort_values("growth_score", ascending=False)

    return emerging


def run_pipeline(config: PipelineConfig) -> None:
    """Run the full bibliometric topic analysis pipeline."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading and cleaning publication metadata...")
    df = load_and_clean_publications(config.input_csv, config.min_year)

    print(f"Loaded {len(df):,} cleaned publications.")

    print("[2/5] Vectorizing publication text with TF-IDF...")
    vectorizer, matrix = vectorize_text(df["text"], config.max_features)

    print("[3/5] Clustering publications...")
    labels, silhouette = cluster_publications(matrix, config.n_clusters)
    df["cluster"] = labels

    print(f"Silhouette score: {silhouette:.4f}")

    print("[4/5] Extracting cluster-level keywords...")
    cluster_summary = extract_cluster_keywords(
        df=df,
        vectorizer=vectorizer,
        matrix=matrix,
        top_n=config.top_keywords_per_cluster,
    )

    print("[5/5] Detecting emerging research topics...")
    emerging_topics = detect_emerging_topics(df, cluster_summary)

    cleaned_path = config.output_dir / "cleaned_publications.csv"
    clustered_path = config.output_dir / "clustered_publications.csv"
    topics_path = config.output_dir / "emerging_topics.csv"

    df.to_csv(cleaned_path, index=False)
    cluster_summary.to_csv(config.output_dir / "cluster_summary.csv", index=False)
    emerging_topics.to_csv(topics_path, index=False)

    print("\nPipeline complete.")
    print(f"Cleaned publications: {cleaned_path}")
    print(f"Clustered publications: {clustered_path}")
    print(f"Emerging topics: {topics_path}")


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Bibliometric NLP pipeline for emerging topic detection."
    )
    parser.add_argument("--input_csv", required=True, help="Input publication CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory for output files.")
    parser.add_argument("--min_year", type=int, default=2015)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--n_clusters", type=int, default=12)
    parser.add_argument("--top_keywords_per_cluster", type=int, default=12)

    args = parser.parse_args()

    return PipelineConfig(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        min_year=args.min_year,
        max_features=args.max_features,
        n_clusters=args.n_clusters,
        top_keywords_per_cluster=args.top_keywords_per_cluster,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
