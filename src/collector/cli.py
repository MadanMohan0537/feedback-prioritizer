"""Command line interface for Pulse's collection layer."""

import argparse
import json
import os

from .connectors import AppleReviewsConnector, FileConnector, GenericJSONConnector, GooglePlayConnector, IntercomConnector, SyntheticConnector, TypeformConnector, ZendeskConnector
from .pipeline import collect
from .storage import FeedbackStore


def parser():
    root = argparse.ArgumentParser(prog="pulse", description="Collect and normalize customer feedback.")
    root.add_argument("--database", default=os.getenv("PULSE_DATABASE", "data/pulse.db"))
    commands = root.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="Initialize the SQLite database")
    initialize.set_defaults(action="init")

    fetch = commands.add_parser("fetch", help="Fetch from a configured source")
    fetch.add_argument("--source", required=True, choices=["synthetic", "file", "typeform", "zendesk", "intercom", "apple", "google-play", "json"])
    fetch.add_argument("--path")
    fetch.add_argument("--url")
    fetch.add_argument("--name", default="generic")
    fetch.add_argument("--items-key", default="items")
    fetch.add_argument("--count", type=int, default=100)
    fetch.add_argument("--no-redact", action="store_true")

    export = commands.add_parser("export", help="Export normalized records as JSONL")
    export.add_argument("--output", required=True)
    export.add_argument("--source", default="")

    stats = commands.add_parser("stats", help="Print collection statistics")
    delete = commands.add_parser("delete-user", help="Delete records for a hashed user ID")
    delete.add_argument("user_id")
    return root


def connector_from(args):
    if args.source == "synthetic":
        return SyntheticConnector(args.count)
    if args.source == "file":
        if not args.path:
            raise SystemExit("--path is required for file imports")
        return FileConnector(args.path)
    if args.source == "json":
        if not args.url:
            raise SystemExit("--url is required for generic JSON")
        return GenericJSONConnector(args.name, args.url, os.getenv("PULSE_API_TOKEN", ""), args.items_key)
    if args.source == "typeform":
        return TypeformConnector(os.environ["TYPEFORM_FORM_ID"], os.environ["TYPEFORM_TOKEN"])
    if args.source == "zendesk":
        return ZendeskConnector(os.environ["ZENDESK_SUBDOMAIN"], os.environ["ZENDESK_TOKEN"])
    if args.source == "intercom":
        return IntercomConnector(os.environ["INTERCOM_TOKEN"])
    if args.source == "apple":
        return AppleReviewsConnector(os.environ["APPLE_APP_ID"], os.environ["APPLE_CONNECT_TOKEN"])
    if args.source == "google-play":
        return GooglePlayConnector(os.environ["GOOGLE_PLAY_PACKAGE"], os.environ["GOOGLE_PLAY_ACCESS_TOKEN"])
    raise SystemExit(f"Unknown source: {args.source}")


def main(argv=None):
    args = parser().parse_args(argv)
    store = FeedbackStore(args.database)
    if args.command == "init":
        print(json.dumps({"database": str(store.path), "status": "ready"}))
    elif args.command == "fetch":
        report = collect(connector_from(args), store, redact=not args.no_redact)
        print(json.dumps(report.to_dict(), indent=2))
    elif args.command == "export":
        print(json.dumps({"exported": store.export_jsonl(args.output, args.source), "output": args.output}))
    elif args.command == "stats":
        print(json.dumps({"records": store.count(), "database": str(store.path)}, indent=2))
    elif args.command == "delete-user":
        print(json.dumps({"deleted": store.delete_user(args.user_id)}))


if __name__ == "__main__":
    main()
