#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator

import pandas as pd
from tqdm import tqdm

from pubchem_client import PubChemClient
from smiles_props import SmilesPropertyCalculator


class StructureLookupRunner:
    def __init__(
        self,
        input_path: str | Path,
        outdir: str | Path,
        smiles_col: str = "SMILES",
        cache_dir: str | Path = "./pubchem_cache",
        use_pubchem_properties: bool = True,
        n_threads: int = 8,
        verbose: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        self.outdir = Path(outdir)
        self.smiles_col = smiles_col
        self.cache_dir = Path(cache_dir)
        self.use_pubchem_properties = use_pubchem_properties
        self.n_threads = max(1, n_threads)
        self.verbose = verbose

        self.outdir.mkdir(parents=True, exist_ok=True)

        self.calc = SmilesPropertyCalculator()
        self.pubchem = PubChemClient(cache_dir=self.cache_dir)

        self._parse_cache: dict[str, Any] = {}
        self._rdkit_cache: dict[str, Dict[str, str]] = {}
        self._evidence_cache: dict[str, Any | None] = {}
        self._cid_props_cache: dict[int, Dict[str, str]] = {}

    def run(self) -> tuple[Path, Path]:
        rows = list(self._read_rows())

        out_tsv = self.outdir / f"{self.input_path.stem}.enriched.tsv"
        out_jsonl = self.outdir / f"{self.input_path.stem}.raw.jsonl"

        prepared = [
            self._prepare_local_row(row)
            for row in tqdm(rows, desc="Preparing rows", unit="row")
        ]

        self._prefetch_pubchem(prepared)

        fieldnames: list[str] = []
        seen_fields: set[str] = set()

        for item in prepared:
            for key in item["out_row"]:
                if key not in seen_fields:
                    seen_fields.add(key)
                    fieldnames.append(key)

            for key in item["pub_props"]:
                if key not in seen_fields:
                    seen_fields.add(key)
                    fieldnames.append(key)

        with (
            out_jsonl.open("w", encoding="utf-8") as json_handle,
            out_tsv.open("w", encoding="utf-8", newline="") as tsv_handle,
        ):
            writer = csv.DictWriter(tsv_handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for item in tqdm(prepared, desc="Writing output", unit="row"):
                out_row = dict(item["out_row"])
                out_row.update(item["pub_props"])

                raw = {
                    "input": item["input"],
                    "parse": asdict(item["parse"]),
                    "rdkit_props": item["rdprops"],
                    "pubchem_evidence": asdict(item["ev"]) if item["ev"] is not None else None,
                    "pubchem_props": item["pub_props"],
                }
                json_handle.write(json.dumps(raw) + "\n")
                writer.writerow(out_row)

        return out_tsv, out_jsonl

    def _prepare_local_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        smi = (row.get(self.smiles_col) or "").strip()

        parse = self._parse_smiles(smi)
        out_row: Dict[str, str] = dict(row)
        out_row["canonical_smiles"] = parse.canonical_smiles
        out_row["inchi"] = parse.inchi
        out_row["inchikey"] = parse.inchikey
        out_row["smiles_ok"] = "1" if parse.ok else "0"
        out_row["smiles_error"] = parse.error

        rdkit_input = parse.canonical_smiles if parse.ok else smi
        rdprops = self._rdkit_props(rdkit_input)
        out_row.update({f"rdkit_{key}": value for key, value in rdprops.items()})

        return {
            "input": row,
            "parse": parse,
            "rdprops": rdprops,
            "out_row": out_row,
            "ev": None,
            "pub_props": {},
        }

    def _prefetch_pubchem(self, prepared: list[Dict[str, Any]]) -> None:
        inchikeys = {
            item["parse"].inchikey
            for item in prepared
            if item["parse"].inchikey
        }

        if inchikeys:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                future_to_inchikey = {
                    executor.submit(self._evidence_by_inchikey, inchikey): inchikey
                    for inchikey in inchikeys
                }

                for future in tqdm(
                    as_completed(future_to_inchikey),
                    total=len(future_to_inchikey),
                    desc="PubChem evidence",
                    unit="compound",
                ):
                    inchikey = future_to_inchikey[future]
                    try:
                        future.result()
                    except Exception as exc:
                        if self.verbose:
                            print(
                                f"WARNING: evidence lookup failed for {inchikey}: {exc}",
                                file=sys.stderr,
                            )

        cids = {
            ev.cid
            for ev in self._evidence_cache.values()
            if ev is not None and getattr(ev, "cid", None) is not None and ev.found_in_pubchem
        }

        if self.use_pubchem_properties and cids:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                future_to_cid = {
                    executor.submit(self._properties_by_cid, cid): cid
                    for cid in cids
                }

                for future in tqdm(
                    as_completed(future_to_cid),
                    total=len(future_to_cid),
                    desc="PubChem properties",
                    unit="compound",
                ):
                    cid = future_to_cid[future]
                    try:
                        future.result()
                    except Exception as exc:
                        if self.verbose:
                            print(
                                f"WARNING: property lookup failed for CID {cid}: {exc}",
                                file=sys.stderr,
                            )

        for item in prepared:
            parse = item["parse"]
            out_row = item["out_row"]

            if not parse.inchikey:
                out_row["pubchem_found"] = "0"
                out_row["pubchem_cid"] = ""
                out_row["pubchem_pubmed_count"] = "0"
                out_row["pubchem_patent_count"] = "0"
                out_row["pubchem_synonyms_sample"] = ""
                item["ev"] = None
                item["pub_props"] = {}
                continue

            ev = self._evidence_cache.get(parse.inchikey)
            item["ev"] = ev

            if ev is None:
                out_row["pubchem_found"] = "0"
                out_row["pubchem_cid"] = ""
                out_row["pubchem_pubmed_count"] = "0"
                out_row["pubchem_patent_count"] = "0"
                out_row["pubchem_synonyms_sample"] = ""
                item["pub_props"] = {}
                continue

            out_row["pubchem_found"] = "1" if ev.found_in_pubchem else "0"
            out_row["pubchem_cid"] = str(ev.cid or "")
            out_row["pubchem_pubmed_count"] = str(ev.pubmed_count)
            out_row["pubchem_patent_count"] = str(ev.patent_count)
            out_row["pubchem_synonyms_sample"] = ev.synonym_sample

            if self.verbose and ev.found_in_pubchem:
                tqdm.write(
                    (
                        f"pubchem_found: {out_row['pubchem_found']} | "
                        f"pubchem_cid: {out_row['pubchem_cid']} | "
                        f"pubchem_pubmed_count: {out_row['pubchem_pubmed_count']} | "
                        f"pubchem_patent_count: {out_row['pubchem_patent_count']} | "
                        f"pubchem_synonyms_sample: {out_row['pubchem_synonyms_sample']}"
                    )
                )

            if self.use_pubchem_properties and ev.cid is not None:
                item["pub_props"] = self._cid_props_cache.get(ev.cid, {})
            else:
                item["pub_props"] = {}

    def _parse_smiles(self, smi: str) -> Any:
        cached = self._parse_cache.get(smi)
        if cached is not None:
            return cached

        parsed = self.calc.parse(smi)
        self._parse_cache[smi] = parsed
        return parsed

    def _rdkit_props(self, smi: str) -> Dict[str, str]:
        cached = self._rdkit_cache.get(smi)
        if cached is not None:
            return cached

        props = self.calc.rdkit_props(smi)
        self._rdkit_cache[smi] = props
        return props

    def _evidence_by_inchikey(self, inchikey: str) -> Any | None:
        if inchikey in self._evidence_cache:
            return self._evidence_cache[inchikey]

        ev = self.pubchem.evidence_by_inchikey(inchikey)
        self._evidence_cache[inchikey] = ev
        return ev

    def _properties_by_cid(self, cid: int) -> Dict[str, str]:
        cached = self._cid_props_cache.get(cid)
        if cached is not None:
            return cached

        props = self.pubchem.properties_by_cid(cid)
        self._cid_props_cache[cid] = props
        return props

    def _read_rows(self) -> Iterator[Dict[str, str]]:
        suffix = self.input_path.suffix.lower()

        if suffix in {".tsv", ".csv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(self.input_path, sep=sep, dtype=str, keep_default_na=False)

            if self.smiles_col not in df.columns:
                raise ValueError(f"Missing column '{self.smiles_col}' in {self.input_path}")

            for row in df.to_dict(orient="records"):
                yield {key: str(value) for key, value in row.items()}
            return

        if suffix in {".txt", ".smi"}:
            with self.input_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    smi = line.strip()
                    if smi:
                        yield {self.smiles_col: smi}
            return

        raise ValueError(f"Unsupported input type: {self.input_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="TSV/CSV with SMILES column, or .txt/.smi with SMILES per line",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--cache-dir", default="./pubchem_cache")
    parser.add_argument("--no-pubchem-props", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    runner = StructureLookupRunner(
        input_path=args.input,
        outdir=args.outdir,
        smiles_col=args.smiles_col,
        cache_dir=args.cache_dir,
        use_pubchem_properties=not args.no_pubchem_props,
        n_threads=args.threads,
        verbose=args.verbose,
    )

    out_tsv, out_jsonl = runner.run()
    print("Wrote:", out_tsv)
    print("Wrote:", out_jsonl)


if __name__ == "__main__":
    main()
