# main.py
import argparse
import json

from api_client import ApiClient
from config import DB_PATH, DEFAULT_BLOCK_WINDOW
from db_utils import get_connection, create_core_tables, get_max_block_in_db
from ingestion import fetch_and_store_blocks
from detection import find_transaction_chain, detect_dos_like_activity


def cmd_ingest(args):
    client = ApiClient()
    conn = get_connection(DB_PATH)
    create_core_tables(conn)

    latest = client.get_latest_block_number()

    if args.start_block is not None and args.end_block is not None:
        start_block = args.start_block
        end_block = args.end_block
    else:
        end_block = latest
        start_block = latest - args.window + 1

    print(f"[INFO] Ingesting blocks {start_block}..{end_block}")
    fetch_and_store_blocks(conn, client, range(start_block, end_block + 1))
    conn.close()


def cmd_detect_chains(args):
    chains = find_transaction_chain(
        db_path=DB_PATH,
        min_length=args.min_length,
        time_window_seconds=args.time_window,
    )
    print(json.dumps(chains, indent=4))


def cmd_detect_dos(args):
    alerts = detect_dos_like_activity(
        db_path=DB_PATH,
        tx_threshold=args.threshold,
        window_minutes=args.window_minutes,
    )
    print(json.dumps(alerts, indent=4))


def build_parser():
    parser = argparse.ArgumentParser(
        description="CS4389 Ethereum Fraud / DoS Detection Toolkit"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest blocks from Sepolia into SQLite")
    p_ingest.add_argument(
        "--start-block",
        type=int,
        help="Start block (inclusive). If omitted, uses latest - window + 1.",
    )
    p_ingest.add_argument(
        "--end-block",
        type=int,
        help="End block (inclusive). If omitted, uses latest.",
    )
    p_ingest.add_argument(
        "--window",
        type=int,
        default=DEFAULT_BLOCK_WINDOW,
        help=f"Number of latest blocks to ingest when start/end not provided (default {DEFAULT_BLOCK_WINDOW})",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # same-value chains
    p_chain = sub.add_parser(
        "detect-chains", help="Detect same-value transaction chains"
    )
    p_chain.add_argument(
        "--min-length",
        type=int,
        default=4,
        help="Minimum length of chain to report (default 4)",
    )
    p_chain.add_argument(
        "--time-window",
        type=int,
        default=3600,
        help="Max seconds between tx in a chain (default 3600 = 1 hour)",
    )
    p_chain.set_defaults(func=cmd_detect_chains)

    # DoS-like high-volume detector
    p_dos = sub.add_parser(
        "detect-dos", help="Detect high-volume transaction bursts (DoS-like behavior)"
    )
    p_dos.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Minimum number of tx in time window to raise alert (default 10)",
    )
    p_dos.add_argument(
        "--window-minutes",
        type=int,
        default=10,
        help="Time window in minutes for bucket (default 10)",
    )
    p_dos.set_defaults(func=cmd_detect_dos)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
