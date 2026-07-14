from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from supabase import create_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--table",
        default=os.getenv("SUPABASE_TABLE", "regulation_chunks"),
    )
    return parser.parse_args()


def to_record(row: dict) -> dict:
    embedding = row["embedding"]
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    metadata = row.get("metadata") or {}
    if hasattr(metadata, "item"):
        metadata = metadata.item()

    return {
        "passage_id": str(row["passage_id"]),
        "title": str(row.get("title", "")),
        "document": str(row.get("document", "")),
        "article": str(row.get("article", "")),
        "content": str(row.get("content", "")),
        "embedding_text": str(row.get("embedding_text", "")),
        "content_hash": str(row.get("content_hash", "")),
        "embedding": [float(value) for value in embedding],
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required."
        )

    dataframe = pd.read_parquet(args.input)
    client = create_client(url, key)

    records = [to_record(row) for row in dataframe.to_dict(orient="records")]

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        (
            client.table(args.table)
            .upsert(batch, on_conflict="passage_id")
            .execute()
        )
        print(f"Imported {min(start + len(batch), len(records))}/{len(records)}")


if __name__ == "__main__":
    main()
