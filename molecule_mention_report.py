#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


USER_AGENT = "Mozilla/5.0 (compatible; molecule-mention-report/1.0)"
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
    "{smiles}/property/CanonicalSMILES,InChIKey,IUPACName/JSON"
)
EUROPEPMC_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    "?format=json&pageSize={page_size}&query={query}"
)
GOOGLE_PATENTS_URL = "https://patents.google.com/?q={query}"


def http_get_json(url: str, timeout: int, retries: int, sleep_s: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"GET JSON failed: {url}: {last_error}") from last_error


def http_get_text(url: str, timeout: int, retries: int, sleep_s: float) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"GET text failed: {url}: {last_error}") from last_error


class RateLimiter:
    def __init__(self, calls_per_second: float):
        self.min_interval = 0.0 if calls_per_second <= 0 else 1.0 / calls_per_second
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delay = self.min_interval - (now - self.last_call)
            if delay > 0:
                time.sleep(delay)
            self.last_call = time.monotonic()


@dataclass
class PubChemIdentity:
    cid: int | None
    inchikey: str | None
    iupac_name: str | None
    canonical_smiles: str | None
    resolve_error: str | None = None


@dataclass
class Mention:
    source_type: str
    database: str
    query: str
    match_mode: str
    confidence: str
    title: str
    year: str
    id: str
    url: str
    authors_or_assignee: str
    journal_or_publication: str
    snippet: str


def read_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if limit is not None:
        rows = rows[:limit]
    return rows


def resolve_pubchem(smiles: str, timeout: int, retries: int, rate_limiter: RateLimiter) -> PubChemIdentity:
    rate_limiter.wait()
    url = PUBCHEM_URL.format(smiles=urllib.parse.quote(smiles, safe=""))
    try:
        payload = http_get_json(url, timeout=timeout, retries=retries, sleep_s=1.0)
        prop = payload["PropertyTable"]["Properties"][0]
        return PubChemIdentity(
            cid=prop.get("CID"),
            inchikey=prop.get("InChIKey"),
            iupac_name=prop.get("IUPACName"),
            canonical_smiles=prop.get("ConnectivitySMILES") or prop.get("CanonicalSMILES"),
            resolve_error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return PubChemIdentity(
            cid=None,
            inchikey=None,
            iupac_name=None,
            canonical_smiles=None,
            resolve_error=str(exc),
        )


def europe_pmc_queries(smiles: str, identity: PubChemIdentity) -> list[tuple[str, str, str]]:
    queries: list[tuple[str, str, str]] = []
    if identity.inchikey:
        queries.append((f'"{identity.inchikey}"', "exact_identifier", "high"))
    if identity.iupac_name:
        queries.append((f'"{identity.iupac_name}"', "exact_name", "medium"))
    canonical = identity.canonical_smiles or smiles
    queries.append((f'"{canonical}"', "exact_smiles", "low"))
    return queries


def search_europe_pmc(
    smiles: str,
    identity: PubChemIdentity,
    page_size: int,
    timeout: int,
    retries: int,
    rate_limiter: RateLimiter,
) -> list[Mention]:
    mentions: list[Mention] = []
    seen: set[tuple[str, str]] = set()
    for query, match_mode, confidence in europe_pmc_queries(smiles, identity):
        rate_limiter.wait()
        url = EUROPEPMC_URL.format(page_size=page_size, query=urllib.parse.quote(query))
        try:
            payload = http_get_json(url, timeout=timeout, retries=retries, sleep_s=1.0)
        except Exception:
            continue
        for item in payload.get("resultList", {}).get("result", []):
            key = ("europepmc", item.get("id", ""))
            if key in seen:
                continue
            seen.add(key)
            authors = item.get("authorString", "")
            journal = item.get("journalTitle", "") or item.get("source", "")
            url = ""
            if item.get("doi"):
                url = f"https://doi.org/{item['doi']}"
            elif item.get("pmid"):
                url = f"https://pubmed.ncbi.nlm.nih.gov/{item['pmid']}/"
            elif item.get("id"):
                url = f"https://europepmc.org/article/{item.get('source', '')}/{item['id']}"
            mentions.append(
                Mention(
                    source_type="paper",
                    database="Europe PMC",
                    query=query,
                    match_mode=match_mode,
                    confidence=confidence,
                    title=item.get("title", ""),
                    year=str(item.get("pubYear", "")),
                    id=item.get("id", ""),
                    url=url,
                    authors_or_assignee=authors,
                    journal_or_publication=journal,
                    snippet=item.get("abstractText", "")[:400],
                )
            )
    return mentions


def patent_queries(smiles: str, identity: PubChemIdentity) -> list[tuple[str, str, str]]:
    queries: list[tuple[str, str, str]] = []
    if identity.inchikey:
        queries.append((f'"{identity.inchikey}"', "exact_identifier", "high"))
    if identity.iupac_name:
        queries.append((f'"{identity.iupac_name}"', "exact_name", "medium"))
    canonical = identity.canonical_smiles or smiles
    queries.append((f'"{canonical}"', "exact_smiles", "low"))
    return queries


def parse_google_patents(html: str, query: str, match_mode: str, confidence: str, limit: int) -> list[Mention]:
    mentions: list[Mention] = []
    seen: set[str] = set()
    pattern = re.compile(r'href="(/patent/([^"?]+)[^"]*)"', re.IGNORECASE)
    titles = re.finditer(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    page_title = ""
    for m in titles:
        page_title = re.sub(r"\s+", " ", m.group(1)).strip()
        break
    for match in pattern.finditer(html):
        patent_path, patent_id = match.groups()
        if patent_id in seen:
            continue
        seen.add(patent_id)
        mentions.append(
            Mention(
                source_type="patent",
                database="Google Patents",
                query=query,
                match_mode=match_mode,
                confidence=confidence,
                title=page_title or patent_id,
                year="",
                id=patent_id,
                url=f"https://patents.google.com{patent_path}",
                authors_or_assignee="",
                journal_or_publication="Google Patents",
                snippet="Search-result page contained this patent identifier. Verify the exact compound mention in the patent body or claims.",
            )
        )
        if len(mentions) >= limit:
            break
    return mentions


def search_google_patents(
    smiles: str,
    identity: PubChemIdentity,
    page_size: int,
    timeout: int,
    retries: int,
    rate_limiter: RateLimiter,
) -> list[Mention]:
    mentions: list[Mention] = []
    seen: set[str] = set()
    for query, match_mode, confidence in patent_queries(smiles, identity):
        rate_limiter.wait()
        url = GOOGLE_PATENTS_URL.format(query=urllib.parse.quote(query))
        try:
            html = http_get_text(url, timeout=timeout, retries=retries, sleep_s=1.0)
        except Exception:
            continue
        for mention in parse_google_patents(html, query, match_mode, confidence, page_size):
            if mention.id in seen:
                continue
            seen.add(mention.id)
            mentions.append(mention)
    return mentions


def dedupe_mentions(mentions: list[Mention]) -> list[Mention]:
    order = {"high": 0, "medium": 1, "low": 2}
    best: dict[tuple[str, str], Mention] = {}
    for mention in mentions:
        key = (mention.database, mention.id or mention.url)
        current = best.get(key)
        if current is None or order[mention.confidence] < order[current.confidence]:
            best[key] = mention
    return sorted(
        best.values(),
        key=lambda x: (x.source_type, order.get(x.confidence, 9), x.year or "", x.title),
    )


def process_row(
    row: dict[str, str],
    args: argparse.Namespace,
    pubchem_rl: RateLimiter,
    epmc_rl: RateLimiter,
    patent_rl: RateLimiter,
) -> dict[str, Any]:
    smiles = row["SMILES"]
    identity = resolve_pubchem(smiles, args.timeout, args.retries, pubchem_rl)
    mentions: list[Mention] = []
    if args.include_papers:
        mentions.extend(search_europe_pmc(smiles, identity, args.per_source_limit, args.timeout, args.retries, epmc_rl))
    if args.include_patents:
        mentions.extend(
            search_google_patents(smiles, identity, args.per_source_limit, args.timeout, args.retries, patent_rl)
        )
    mentions = dedupe_mentions(mentions)
    return {
        "input": row,
        "identity": asdict(identity),
        "mentions": [asdict(m) for m in mentions],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = tsv_fields()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in records:
            write_tsv_record(writer, record)


def tsv_fields() -> list[str]:
    return [
        "Label",
        "InputSMILES",
        "CID",
        "InChIKey",
        "IUPACName",
        "ResolveError",
        "MentionCount",
        "SourceType",
        "Database",
        "Query",
        "MatchMode",
        "Confidence",
        "Title",
        "Year",
        "ReferenceID",
        "URL",
        "AuthorsOrAssignee",
        "JournalOrPublication",
        "Snippet",
    ]


def _tsv_base_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "Label": record["input"].get("Label", ""),
        "InputSMILES": record["input"].get("SMILES", ""),
        "CID": record["identity"].get("cid") or "",
        "InChIKey": record["identity"].get("inchikey") or "",
        "IUPACName": record["identity"].get("iupac_name") or "",
        "ResolveError": record["identity"].get("resolve_error") or "",
        "MentionCount": len(record["mentions"]),
    }


def write_tsv_record(writer: csv.DictWriter, record: dict[str, Any]) -> None:
    base = _tsv_base_record(record)
    if not record["mentions"]:
        writer.writerow(base)
        return
    for mention in record["mentions"]:
        writer.writerow(
            {
                **base,
                "SourceType": mention.get("source_type", ""),
                "Database": mention.get("database", ""),
                "Query": mention.get("query", ""),
                "MatchMode": mention.get("match_mode", ""),
                "Confidence": mention.get("confidence", ""),
                "Title": mention.get("title", ""),
                "Year": mention.get("year", ""),
                "ReferenceID": mention.get("id", ""),
                "URL": mention.get("url", ""),
                "AuthorsOrAssignee": mention.get("authors_or_assignee", ""),
                "JournalOrPublication": mention.get("journal_or_publication", ""),
                "Snippet": mention.get("snippet", ""),
            }
        )


def summarize_records(records: list[dict[str, Any]], include_papers: bool, include_patents: bool) -> str:
    database_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    rows_with_mentions = 0
    resolve_errors = 0

    for record in records:
        mentions = record.get("mentions", [])
        if mentions:
            rows_with_mentions += 1
        if record.get("identity", {}).get("resolve_error"):
            resolve_errors += 1
        for mention in mentions:
            database = mention.get("database", "Unknown") or "Unknown"
            source_type = mention.get("source_type", "unknown") or "unknown"
            database_counts[database] = database_counts.get(database, 0) + 1
            source_counts[source_type] = source_counts.get(source_type, 0) + 1

    looked_into = ["PubChem"]
    if include_papers:
        looked_into.append("Europe PMC")
    if include_patents:
        looked_into.append("Google Patents")

    lines = [
        "Molecule mention processing summary",
        f"Total input rows: {len(records)}",
        f"Rows with mentions: {rows_with_mentions}",
        f"Rows with PubChem resolve errors: {resolve_errors}",
        f"Databases looked into: {', '.join(looked_into)}",
    ]

    if source_counts:
        lines.append("Mentions by source type:")
        for source_type in sorted(source_counts):
            lines.append(f"- {source_type}: {source_counts[source_type]}")

    if database_counts:
        lines.append("Mentions by database:")
        for database in sorted(database_counts):
            lines.append(f"- {database}: {database_counts[database]}")
    else:
        lines.append("Mentions by database: none")

    return "\n".join(lines) + "\n"


def write_summary(path: Path, records: list[dict[str, Any]], include_papers: bool, include_patents: bool) -> None:
    path.write_text(
        summarize_records(records, include_papers=include_papers, include_patents=include_patents),
        encoding="utf-8",
    )


def _compact_text(value: str, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def log_found_mentions(record: dict[str, Any], index: int, total: int) -> None:
    mentions = record.get("mentions", [])
    if not mentions:
        return

    for mention in mentions:
        title = _compact_text(mention.get("title", "") or "-", limit=120)
        year = mention.get("year", "") or "-"
        database = mention.get("database", "") or "Unknown"
        source_type = mention.get("source_type", "") or "unknown"
        confidence = mention.get("confidence", "") or "-"
        match_mode = mention.get("match_mode", "") or "-"
        reference_id = mention.get("id", "") or "-"
        query = _compact_text(mention.get("query", "") or "-", limit=80)
        print(
            f"[{index}/{total}] found source={source_type} db={database} confidence={confidence} "
            f"match={match_mode} year={year} id={reference_id} query={query} title={title}",
            file=sys.stderr,
        )


def log_record_result(record: dict[str, Any], index: int, total: int) -> None:
    label = record.get("input", {}).get("Label", "") or "-"
    smiles = record.get("input", {}).get("SMILES", "") or "-"
    identity = record.get("identity", {})
    cid = identity.get("cid")
    resolve_error = identity.get("resolve_error")
    mentions = record.get("mentions", [])

    database_counts: dict[str, int] = {}
    for mention in mentions:
        database = mention.get("database", "Unknown") or "Unknown"
        database_counts[database] = database_counts.get(database, 0) + 1

    if database_counts:
        found_summary = ", ".join(f"{name}={database_counts[name]}" for name in sorted(database_counts))
        print(
            f"[{index}/{total}] info label={label} smiles={smiles} pubchem_cid={cid or '-'} found={found_summary}",
            file=sys.stderr,
        )
    elif resolve_error:
        print(
            f"[{index}/{total}] info label={label} smiles={smiles} pubchem_error={resolve_error}",
            file=sys.stderr,
        )
    else:
        print(
            f"[{index}/{total}] info label={label} smiles={smiles} pubchem_cid={cid or '-'} found=none",
            file=sys.stderr,
        )
    log_found_mentions(record, index, total)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve SMILES from a TSV and build a paper/patent mention report."
    )
    parser.add_argument("--input", required=True, help="Input TSV path with a SMILES column.")
    parser.add_argument("--out-prefix", default="mention_report", help="Prefix for output files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument("--workers", type=int, default=8, help="Thread count across molecules.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per request.")
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=5,
        help="Maximum mentions to keep per source/query combination.",
    )
    parser.add_argument(
        "--pubchem-rps",
        type=float,
        default=3.0,
        help="PubChem request rate limit in requests per second.",
    )
    parser.add_argument(
        "--europepmc-rps",
        type=float,
        default=5.0,
        help="Europe PMC request rate limit in requests per second.",
    )
    parser.add_argument(
        "--patent-rps",
        type=float,
        default=1.0,
        help="Google Patents request rate limit in requests per second.",
    )
    parser.add_argument("--no-papers", action="store_true", help="Skip Europe PMC paper search.")
    parser.add_argument("--no-patents", action="store_true", help="Skip Google Patents search.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.include_papers = not args.no_papers
    args.include_patents = not args.no_patents

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    rows = read_rows(input_path, args.limit)
    if not rows:
        print("No rows found.", file=sys.stderr)
        return 2
    if "SMILES" not in rows[0]:
        print("Input TSV must contain a SMILES column.", file=sys.stderr)
        return 2

    pubchem_rl = RateLimiter(args.pubchem_rps)
    epmc_rl = RateLimiter(args.europepmc_rps)
    patent_rl = RateLimiter(args.patent_rps)

    records: list[dict[str, Any]] = []
    prefix = Path(args.out_prefix)
    jsonl_path = prefix.with_suffix(".jsonl")
    tsv_path = prefix.with_suffix(".tsv")
    summary_path = prefix.with_name(f"{prefix.name}_summary").with_suffix(".txt")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_row, row, args, pubchem_rl, epmc_rl, patent_rl)
            for row in rows
        ]
        with jsonl_path.open("w", encoding="utf-8") as jsonl_handle, tsv_path.open(
            "w", newline="", encoding="utf-8"
        ) as tsv_handle:
            tsv_writer = csv.DictWriter(tsv_handle, fieldnames=tsv_fields(), delimiter="\t")
            tsv_writer.writeheader()
            tsv_handle.flush()
            write_summary(summary_path, records, include_papers=args.include_papers, include_patents=args.include_patents)

            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "input": {"SMILES": "", "Label": ""},
                        "identity": asdict(PubChemIdentity(None, None, None, None, str(exc))),
                        "mentions": [],
                    }
                records.append(record)
                append_jsonl_record(jsonl_handle, record)
                write_tsv_record(tsv_writer, record)
                tsv_handle.flush()
                write_summary(
                    summary_path, records, include_papers=args.include_papers, include_patents=args.include_patents
                )
                log_record_result(record, index, len(rows))
                print(f"[{index}/{len(rows)}] processed", file=sys.stderr)

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {tsv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
