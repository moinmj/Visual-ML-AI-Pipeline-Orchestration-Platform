# """
# Registers every tabular Ozone file that has no dataset yet, so each one gets a dataset id.

# It only uses the two existing endpoints:
#   GET  /api/v1/datasets/ozone/browse      (finds files with dataset_id == null and can_register == true)
#   POST /api/v1/datasets/ozone/register    (registers one file; no data is copied)

# Usage (PowerShell / bash):
#   python register_all_ozone.py --base-url http://localhost:8000 --token <BEARER_TOKEN> --dry-run
#   python register_all_ozone.py --base-url http://localhost:8000 --token <BEARER_TOKEN>

# Optional:
#   --skip-pattern "_size_test"             skip files whose name contains any of these (comma separated)
#   --max-size-mb 50                        skip files bigger than this (registering reads the whole file)
#   --workers 3                             parallel register calls (default 3)

# The token can also come from the environment variable DATAREPT_TOKEN.
# Results are written to register_results.csv (key, dataset_id, status).
# """
# import argparse
# import csv
# import os
# import sys
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import requests

# TABULAR = {"csv", "xlsx", "xls", "json", "parquet"}


# def parse_args():
#     p = argparse.ArgumentParser(description="Register all unregistered tabular Ozone files as datasets.")
#     p.add_argument("--base-url", default="http://localhost:8000", help="API base URL, without /api/v1")
#     p.add_argument("--token", default=os.environ.get("DATAREPT_TOKEN", ""), help="Bearer token")
#     p.add_argument("--scope", default="ml_workflow", choices=["ml_workflow", "all_tenant_data", "bucket", "landing"])
#     p.add_argument("--folder", default="datasets", choices=["all", "datasets", "unstructured"])
#     p.add_argument("--skip-pattern", default="", help="comma separated name parts to skip")
#     p.add_argument("--max-size-mb", type=float, default=0, help="skip files above this size (0 = no limit)")
#     p.add_argument("--workers", type=int, default=3)
#     p.add_argument("--dry-run", action="store_true", help="only list what would be registered")
#     p.add_argument("--out", default="register_results.csv")
#     return p.parse_args()


# def main():
#     args = parse_args()
#     if not args.token:
#         sys.exit("No token. Pass --token or set DATAREPT_TOKEN.")

#     base = args.base_url.rstrip("/") + "/api/v1/datasets"
#     headers = {"Authorization": f"Bearer {args.token}"}
#     skip_parts = [s.strip().lower() for s in args.skip_pattern.split(",") if s.strip()]

#     # 1. Browse (tabular files only)
#     r = requests.get(
#         f"{base}/ozone/browse",
#         headers=headers,
#         params={"scope": args.scope, "folder": args.folder, "media_type": "tabular", "limit": 5000},
#         timeout=120,
#     )
#     r.raise_for_status()
#     items = r.json()
#     print(f"Browse returned {len(items)} tabular items.")

#     already, todo, skipped = [], [], []
#     for it in items:
#         if it.get("dataset_id"):
#             already.append(it)
#         elif not it.get("can_register") or it.get("file_format") not in TABULAR:
#             skipped.append((it, "not registerable"))
#         elif any(part in (it.get("file_name") or "").lower() for part in skip_parts):
#             skipped.append((it, "skip-pattern"))
#         elif args.max_size_mb and (it.get("file_size_bytes") or 0) > args.max_size_mb * 1024 * 1024:
#             skipped.append((it, "too large"))
#         else:
#             todo.append(it)

#     print(f"Already registered: {len(already)} | To register: {len(todo)} | Skipped: {len(skipped)}")
#     for it, why in skipped:
#         print(f"  skipped ({why}): {it.get('file_name')}")

#     rows = [(it["key"], it["dataset_id"], "already_registered") for it in already]
#     rows += [(it["key"], "", f"skipped:{why}") for it, why in skipped]

#     if args.dry_run:
#         print("\nDry run, would register:")
#         for it in todo:
#             print(f"  {it['key']}")
#         return

#     # 2. Register one by one (no name/description sent, so the name defaults to the file name)
#     def register(it):
#         resp = requests.post(f"{base}/ozone/register", headers=headers, data={"ozone_key": it["key"]}, timeout=300)
#         if resp.status_code in (200, 201):
#             return it["key"], resp.json().get("id", ""), "registered"
#         try:
#             detail = resp.json()
#         except Exception:
#             detail = resp.text[:200]
#         return it["key"], "", f"failed:{resp.status_code}:{str(detail)[:120]}"

#     done = 0
#     with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
#         futures = [pool.submit(register, it) for it in todo]
#         for f in as_completed(futures):
#             key, ds_id, status = f.result()
#             rows.append((key, ds_id, status))
#             done += 1
#             print(f"[{done}/{len(todo)}] {status:<12} {ds_id or '-'}  {key}")

#     with open(args.out, "w", newline="", encoding="utf-8") as fh:
#         w = csv.writer(fh)
#         w.writerow(["key", "dataset_id", "status"])
#         w.writerows(rows)

#     ok = sum(1 for _, _, s in rows if s == "registered")
#     bad = sum(1 for _, _, s in rows if s.startswith("failed"))
#     print(f"\nDone. Registered: {ok} | Failed: {bad} | Details in {args.out}")


# if __name__ == "__main__":
#     main()

"""
Registers every tabular Ozone file that has no dataset yet, so each one gets a dataset id.

It only uses the two existing endpoints:
  GET  /api/v1/datasets/ozone/browse      (finds files with dataset_id == null and can_register == true)
  POST /api/v1/datasets/ozone/register    (registers one file; no data is copied)

Usage (PowerShell / bash):
  python register_all_ozone.py --base-url http://localhost:8000 --token <BEARER_TOKEN> --scope landing --dry-run
  python register_all_ozone.py --base-url http://localhost:8000 --token <BEARER_TOKEN> --scope landing

Optional:
  --skip-pattern "_size_test"             skip files whose name contains any of these (comma separated)
  --max-size-mb 50                        skip files bigger than this (registering reads the whole file)
  --workers 3                             parallel register calls (default 3)
  --timeout 900                           browse timeout in seconds (default 900)
  --limit 5000                            max items returned by browse (default 5000)

The token can also come from the environment variable DATAREPT_TOKEN.
Results are written to register_results.csv (key, dataset_id, status).
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TABULAR = {"csv", "xlsx", "xls", "json", "parquet"}


def parse_args():
    p = argparse.ArgumentParser(description="Register all unregistered tabular Ozone files as datasets.")
    p.add_argument("--base-url", default="http://localhost:8000", help="API base URL, without /api/v1")
    p.add_argument("--token", default=os.environ.get("DATAREPT_TOKEN", ""), help="Bearer token")
    p.add_argument("--scope", default="ml_workflow", choices=["ml_workflow", "all_tenant_data", "bucket", "landing"])
    p.add_argument("--folder", default="datasets", choices=["all","db", "datasets", "unstructured"])
    p.add_argument("--skip-pattern", default="", help="comma separated name parts to skip")
    p.add_argument("--max-size-mb", type=float, default=0, help="skip files above this size (0 = no limit)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--timeout", type=int, default=900, help="browse timeout in seconds")
    p.add_argument("--limit", type=int, default=5000, help="max items returned by browse")
    p.add_argument("--dry-run", action="store_true", help="only list what would be registered")
    p.add_argument("--out", default="register_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.token:
        sys.exit("No token. Pass --token or set DATAREPT_TOKEN.")

    base = args.base_url.rstrip("/") + "/api/v1/datasets"
    headers = {"Authorization": f"Bearer {args.token}"}
    skip_parts = [s.strip().lower() for s in args.skip_pattern.split(",") if s.strip()]

    # 1. Browse (tabular files only)
    print(f"Browsing scope={args.scope} folder={args.folder} (timeout {args.timeout}s)...")
    r = requests.get(
        f"{base}/ozone/browse",
        headers=headers,
        params={"scope": args.scope, "folder": args.folder, "media_type": "tabular", "limit": args.limit},
        timeout=args.timeout,
    )
    r.raise_for_status()
    items = r.json()
    print(f"Browse returned {len(items)} tabular items.")

    already, todo, skipped = [], [], []
    for it in items:
        if it.get("dataset_id"):
            already.append(it)
        elif not it.get("can_register") or it.get("file_format") not in TABULAR:
            skipped.append((it, "not registerable"))
        elif any(part in (it.get("file_name") or "").lower() for part in skip_parts):
            skipped.append((it, "skip-pattern"))
        elif args.max_size_mb and (it.get("file_size_bytes") or 0) > args.max_size_mb * 1024 * 1024:
            skipped.append((it, "too large"))
        else:
            todo.append(it)

    print(f"Already registered: {len(already)} | To register: {len(todo)} | Skipped: {len(skipped)}")
    for it, why in skipped:
        print(f"  skipped ({why}): {it.get('file_name')}")

    rows = [(it["key"], it["dataset_id"], "already_registered") for it in already]
    rows += [(it["key"], "", f"skipped:{why}") for it, why in skipped]

    if args.dry_run:
        print("\nDry run, would register:")
        for it in todo:
            print(f"  {it['key']}")
        return

    # 2. Register one by one (no name/description sent, so the name defaults to the file name)
    def register(it):
        resp = requests.post(f"{base}/ozone/register", headers=headers, data={"ozone_key": it["key"]}, timeout=300)
        if resp.status_code in (200, 201):
            return it["key"], resp.json().get("id", ""), "registered"
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:200]
        return it["key"], "", f"failed:{resp.status_code}:{str(detail)[:120]}"

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(register, it) for it in todo]
        for f in as_completed(futures):
            key, ds_id, status = f.result()
            rows.append((key, ds_id, status))
            done += 1
            print(f"[{done}/{len(todo)}] {status:<12} {ds_id or '-'}  {key}")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "dataset_id", "status"])
        w.writerows(rows)

    ok = sum(1 for _, _, s in rows if s == "registered")
    bad = sum(1 for _, _, s in rows if s.startswith("failed"))
    print(f"\nDone. Registered: {ok} | Failed: {bad} | Details in {args.out}")


if __name__ == "__main__":
    main()