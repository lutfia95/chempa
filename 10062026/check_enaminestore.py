#!/usr/bin/env python3
"""Check table S3 catalog IDs against the official Enamine Store search API."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import sys
import time
import uuid
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
except ImportError as exc:
    raise SystemExit("Missing dependency: requests. Install it with 'pip install requests'.") from exc

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: cryptography. Install it with 'pip install cryptography'."
    ) from exc


BASE_URL = "https://enaminestore.com"
SEARCH_URL = f"{BASE_URL}/api/v2/catalog/search/by/codes"
DELAYED_RESULT_URL = f"{BASE_URL}/api/v2/delayed-results/by/uuid/{{result_uuid}}"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/137.0 Safari/537.36"
)


class EnamineLookupError(RuntimeError):
    """Raised when a batch cannot be classified reliably."""


def batched(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def read_input(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"smiles", "catalog_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required TSV columns: {', '.join(sorted(missing))}")

        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader, start=1):
            if limit is not None and index > limit:
                break
            smiles = (row.get("smiles") or "").strip()
            catalog_id = (row.get("catalog_id") or "").strip()
            if not catalog_id:
                raise ValueError(f"Empty catalog_id at TSV data row {index}")
            rows.append({"smiles": smiles, "catalog_id": catalog_id})
    return rows


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid cache format: {path}")

    cache: dict[str, str] = {}
    for catalog_id, status in data.items():
        normalized_status = str(status).lower()
        if normalized_status in {"yes", "no"}:
            cache[str(catalog_id).upper()] = normalized_status
    return cache


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(cache.items())), handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_output(
    path: Path,
    rows: list[dict[str, str]],
    cache: dict[str, str],
) -> None:
    unresolved = [
        row["catalog_id"]
        for row in rows
        if row["catalog_id"].upper() not in cache
    ]
    if unresolved:
        raise EnamineLookupError(
            f"Cannot write final output: {len(unresolved)} catalog IDs remain unresolved"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["smiles", "catalog_id", "found_in_enaminestore"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "smiles": row["smiles"],
                    "catalog_id": row["catalog_id"],
                    "found_in_enaminestore": cache[row["catalog_id"].upper()],
                }
            )
    os.replace(temporary, path)


class EnamineStoreClient:
    def __init__(
        self,
        *,
        timeout: float,
        retries: int,
        delay: float,
        delayed_result_timeout: float,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.delayed_result_timeout = delayed_result_timeout
        self.client_uuid = str(uuid.uuid4())
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json, text/plain, */*",
                "Origin": BASE_URL,
                "Referer": f"{BASE_URL}/",
                "client-application-uuid": self.client_uuid,
            }
        )
        self.decryption_key = self._load_decryption_key()

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        last_error = ""
        for attempt in range(1, self.retries + 1):
            response: requests.Response | None = None
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs,
                )
                if response.status_code == 200:
                    return response
                if response.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                    raise EnamineLookupError(
                        f"HTTP {response.status_code} from {url}: {response.text[:300]}"
                    )
                last_error = f"HTTP {response.status_code}"
            except (requests.RequestException, EnamineLookupError) as exc:
                last_error = str(exc)

            if attempt < self.retries:
                if response is not None and response.status_code == 429:
                    wait = max(
                        self._retry_after_seconds(response),
                        15.0 * attempt,
                    )
                    print(
                        f"[WARN] Enamine Store rate limit; waiting {wait:.1f} seconds "
                        f"before retry {attempt + 1}/{self.retries}.",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    wait = self.delay * (2 ** (attempt - 1))
                wait += random.uniform(0, 0.5)
                time.sleep(wait)
        raise EnamineLookupError(
            f"Request failed after {self.retries} attempts: {last_error}"
        )

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float:
        value = (response.headers.get("Retry-After") or "").strip()
        if not value:
            return 0.0
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                return max(0.0, retry_at.timestamp() - time.time())
            except (TypeError, ValueError):
                return 0.0

    def _load_decryption_key(self) -> bytes:
        homepage = self._request("GET", f"{BASE_URL}/").text
        bundle_match = re.search(
            r'<script[^>]+src="(/static/js/main\.[^"]+\.js)"',
            homepage,
        )
        if not bundle_match:
            raise EnamineLookupError("Could not locate the Enamine Store JavaScript bundle")

        bundle_url = f"{BASE_URL}{bundle_match.group(1)}"
        bundle = self._request("GET", bundle_url).text
        key_match = re.search(r'REACT_APP_DECRYPTION_KEY:"([^"]+)"', bundle)
        if not key_match:
            raise EnamineLookupError("Could not locate the storefront response key")

        key = key_match.group(1).encode("utf-8")
        if len(key) not in {16, 24, 32}:
            raise EnamineLookupError(
                f"Unexpected storefront response-key length: {len(key)}"
            )
        return key

    def _decode_response(self, response: requests.Response) -> dict[str, Any]:
        text = response.text.strip()
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                return parsed
        except requests.JSONDecodeError:
            pass

        try:
            encrypted = base64.b64decode(text, validate=True)
            if len(encrypted) <= 32:
                raise ValueError("encrypted response is too short")
            plaintext = AESGCM(self.decryption_key).decrypt(
                encrypted[:16],
                encrypted[16:],
                None,
            )
            parsed = json.loads(plaintext)
        except Exception as exc:
            raise EnamineLookupError(
                f"Could not decode Enamine Store response: {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise EnamineLookupError("Enamine Store returned a non-object response")
        return parsed

    def _resolve_delayed_result(self, result_uuid: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.delayed_result_timeout
        url = DELAYED_RESULT_URL.format(result_uuid=result_uuid)
        while time.monotonic() < deadline:
            response = self._request("GET", url)
            payload = self._decode_response(response)
            if payload.get("success"):
                delayed_result = payload.get("delayedResult")
                if isinstance(delayed_result, str):
                    parsed = json.loads(delayed_result)
                    if isinstance(parsed, dict):
                        return parsed
                if isinstance(delayed_result, dict):
                    return delayed_result
                raise EnamineLookupError("Delayed result completed without a JSON result")
            time.sleep(max(0.5, self.delay))
        raise EnamineLookupError(
            f"Timed out waiting for delayed Enamine result {result_uuid}"
        )

    def lookup_codes(self, catalog_ids: list[str]) -> dict[str, str]:
        if not catalog_ids:
            return {}

        response = self._request(
            "POST",
            SEARCH_URL,
            json={
                "compounds": " ".join(catalog_ids),
                "currency": "USD",
                "criterias": {},
            },
        )
        payload = self._decode_response(response)
        delayed_uuid = payload.get("delayedResultUuid")
        if delayed_uuid:
            payload = self._resolve_delayed_result(str(delayed_uuid))

        requested = {catalog_id.upper(): catalog_id for catalog_id in catalog_ids}
        statuses: dict[str, str] = {}

        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            candidates: list[str] = []
            product = result.get("product")
            if isinstance(product, dict) and product.get("code"):
                candidates.append(str(product["code"]))
            found_by = result.get("foundBy")
            if isinstance(found_by, dict):
                for raw in found_by.get("raw") or []:
                    if isinstance(raw, dict) and raw.get("compound"):
                        candidates.append(str(raw["compound"]))
            for candidate in candidates:
                normalized = candidate.upper()
                if normalized in requested:
                    statuses[normalized] = "yes"

        for not_found in payload.get("notFound") or []:
            if not isinstance(not_found, dict):
                continue
            for raw in not_found.get("raw") or []:
                if not isinstance(raw, dict) or not raw.get("compound"):
                    continue
                normalized = str(raw["compound"]).upper()
                if normalized in requested and normalized not in statuses:
                    statuses[normalized] = "no"

        unresolved = sorted(set(requested).difference(statuses))
        if unresolved:
            sample = ", ".join(requested[item] for item in unresolved[:5])
            raise EnamineLookupError(
                f"Enamine response did not classify {len(unresolved)} IDs; sample: {sample}"
            )
        return statuses


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Read table_s3.tsv, search catalog_id values in Enamine Store, and write "
            "smiles/catalog_id/found_in_enaminestore."
        )
    )
    parser.add_argument(
        "input_tsv",
        nargs="?",
        type=Path,
        default=script_dir / "converted" / "table_s3.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "converted" / "table_s3_enaminestore.tsv",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=script_dir / "converted" / "table_s3_enaminestore_cache.json",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--delayed-result-timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, help="Only process the first N rows")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached yes/no results and query them again",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    input_path = args.input_tsv.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    cache_path = args.cache.expanduser().resolve()

    if not input_path.is_file():
        print(f"Input TSV not found: {input_path}", file=sys.stderr)
        return 2

    rows = read_input(input_path, args.limit)
    cache = {} if args.refresh else load_cache(cache_path)
    unique_ids = list(dict.fromkeys(row["catalog_id"].upper() for row in rows))
    pending = [catalog_id for catalog_id in unique_ids if catalog_id not in cache]

    print(
        f"[INFO] Loaded {len(rows)} rows and {len(unique_ids)} unique catalog IDs; "
        f"{len(pending)} require lookup.",
        flush=True,
    )

    if pending:
        client = EnamineStoreClient(
            timeout=args.timeout,
            retries=args.retries,
            delay=args.delay,
            delayed_result_timeout=args.delayed_result_timeout,
        )
        batches = list(batched(pending, args.batch_size))
        for batch_number, batch in enumerate(batches, start=1):
            statuses = client.lookup_codes(batch)
            cache.update(statuses)
            save_cache(cache_path, cache)
            yes_count = sum(status == "yes" for status in statuses.values())
            no_count = len(statuses) - yes_count
            print(
                f"[OK] Batch {batch_number}/{len(batches)}: "
                f"{yes_count} yes, {no_count} no",
                flush=True,
            )
            if batch_number < len(batches):
                time.sleep(args.delay)

    write_output(output_path, rows, cache)
    yes_total = sum(
        cache[row["catalog_id"].upper()] == "yes"
        for row in rows
    )
    print(
        f"[DONE] Wrote {len(rows)} rows to {output_path}: "
        f"{yes_total} yes, {len(rows) - yes_total} no.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EnamineLookupError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
