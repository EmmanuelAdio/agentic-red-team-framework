from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from pymongo import MongoClient


TRACE_COLLECTIONS = {"query_traces", "rag_responses"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear or prune MongoDB data used by this project."
    )
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGODB_DB_NAME", "agentic_red_team_baseline"))
    parser.add_argument(
        "--prune-versions",
        type=int,
        default=None,
        help="Delete this many older corpus versions (documents/chunks/corpus_versions only).",
    )
    parser.add_argument(
        "--keep-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When pruning versions, keep the newest corpus version.",
    )
    parser.add_argument(
        "--drop-database",
        action="store_true",
        help="Drop the entire database. If omitted, each collection is deleted in place.",
    )
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="When clearing collections, also delete query_traces and rag_responses.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation flag to execute destructive operations.",
    )
    return parser.parse_args()


def prune_versions(db, prune_versions: int, keep_latest: bool) -> None:
    if prune_versions < 1:
        raise SystemExit("--prune-versions must be >= 1")

    versions = list(db["corpus_versions"].find({}, {"_id": 0}).sort("created_at", -1))
    eligible = versions[1:] if keep_latest else versions
    to_delete = eligible[-prune_versions:] if eligible else []
    delete_versions = [str(row["corpus_version"]) for row in to_delete]

    deleted_documents = int(db["documents"].delete_many({"corpus_version": {"$in": delete_versions}}).deleted_count)
    deleted_chunks = int(db["chunks"].delete_many({"corpus_version": {"$in": delete_versions}}).deleted_count)
    deleted_versions_count = int(db["corpus_versions"].delete_many({"corpus_version": {"$in": delete_versions}}).deleted_count)

    remaining_versions = [str(value) for value in db["corpus_versions"].distinct("corpus_version") if value is not None]
    if remaining_versions:
        orphan_doc_query = {"corpus_version": {"$nin": remaining_versions}}
        orphan_chunk_query = {"corpus_version": {"$nin": remaining_versions}}
    else:
        orphan_doc_query = {}
        orphan_chunk_query = {}

    orphan_documents_deleted = int(db["documents"].delete_many(orphan_doc_query).deleted_count)
    orphan_chunks_deleted = int(db["chunks"].delete_many(orphan_chunk_query).deleted_count)

    print(
        "Pruned corpus versions "
        f"(requested={prune_versions}, deleted={len(delete_versions)}). "
        f"documents_deleted={deleted_documents}, chunks_deleted={deleted_chunks}, "
        f"version_rows_deleted={deleted_versions_count}, "
        f"orphan_documents_deleted={orphan_documents_deleted}, "
        f"orphan_chunks_deleted={orphan_chunks_deleted}, "
        f"deleted_versions={delete_versions}"
    )


def clear_collections(db, include_traces: bool) -> None:
    collection_names = db.list_collection_names()
    if not include_traces:
        collection_names = [name for name in collection_names if name not in TRACE_COLLECTIONS]

    deleted_total = 0
    for name in collection_names:
        result = db[name].delete_many({})
        deleted_total += int(result.deleted_count)

    trace_msg = "included" if include_traces else "preserved"
    print(
        f"Cleared {len(collection_names)} collections in {db.name}. "
        f"Deleted documents: {deleted_total}. Trace collections {trace_msg}."
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.yes:
        raise SystemExit("Refusing to clear database without --yes confirmation flag.")

    client = MongoClient(args.mongo_uri)
    try:
        if args.drop_database:
            client.drop_database(args.db_name)
            print(f"Dropped MongoDB database: {args.db_name}")
            return

        db = client[args.db_name]
        if args.prune_versions is not None:
            prune_versions(db=db, prune_versions=args.prune_versions, keep_latest=args.keep_latest)
            return

        clear_collections(db=db, include_traces=args.include_traces)
    finally:
        client.close()


if __name__ == "__main__":
    main()
