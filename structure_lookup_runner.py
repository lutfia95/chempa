from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

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
    ) -> None:
        self.input_path = Path(input_path)
        self.outdir = Path(outdir)
        self.smiles_col = smiles_col
        self.cache_dir = Path(cache_dir)
        self.use_pubchem_properties = use_pubchem_properties

        self.outdir.mkdir(parents=True, exist_ok=True)

        self.calc = SmilesPropertyCalculator()
        self.pubchem = PubChemClient(cache_dir=self.cache_dir)

    def run(self) -> tuple[Path, Path]:
        rows = self._read_rows()

        out_tsv = self.outdir / f"{self.input_path.stem}.enriched.tsv"
        out_jsonl = self.outdir / f"{self.input_path.stem}.raw.jsonl"

        enriched: List[Dict[str, str]] = []

        with out_jsonl.open("w", encoding="utf-8") as fj:
            for r in rows:
                smi = (r.get(self.smiles_col) or "").strip()
                parse = self.calc.parse(smi)

                out_row: Dict[str, str] = dict(r)
                out_row["canonical_smiles"] = parse.canonical_smiles
                out_row["inchi"] = parse.inchi
                out_row["inchikey"] = parse.inchikey
                out_row["smiles_ok"] = "1" if parse.ok else "0"
                out_row["smiles_error"] = parse.error

                rdprops = self.calc.rdkit_props(parse.canonical_smiles if parse.ok else smi)
                out_row.update({f"rdkit_{k}": v for k, v in rdprops.items()})

                ev = self.pubchem.evidence_by_inchikey(parse.inchikey) if parse.inchikey else None
                if ev is None:
                    out_row["pubchem_found"] = "0"
                    out_row["pubchem_cid"] = ""
                    out_row["pubchem_pubmed_count"] = "0"
                    out_row["pubchem_patent_count"] = "0"
                    out_row["pubchem_synonyms_sample"] = ""
                    pub_props: Dict[str, str] = {}
                else:
                    out_row["pubchem_found"] = "1" if ev.found_in_pubchem else "0"
                    out_row["pubchem_cid"] = str(ev.cid or "")
                    out_row["pubchem_pubmed_count"] = str(ev.pubmed_count)
                    out_row["pubchem_patent_count"] = str(ev.patent_count)
                    out_row["pubchem_synonyms_sample"] = ev.synonym_sample

                    pub_props = {}
                    if self.use_pubchem_properties and ev.cid is not None:
                        pub_props = self.pubchem.properties_by_cid(ev.cid)
                        out_row.update(pub_props)

                raw = {
                    "input": r,
                    "parse": asdict(parse),
                    "rdkit_props": rdprops,
                    "pubchem_evidence": asdict(ev) if ev is not None else None,
                    "pubchem_props": pub_props,
                }
                fj.write(json.dumps(raw) + "\n")

                enriched.append(out_row)

        df = pd.DataFrame(enriched)
        df.to_csv(out_tsv, sep="\t", index=False)

        return out_tsv, out_jsonl

    def _read_rows(self) -> Iterator[Dict[str, str]]:
        if self.input_path.suffix.lower() in {".tsv", ".csv"}:
            sep = "\t" if self.input_path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(self.input_path, sep=sep, dtype=str, keep_default_na=False)
            if self.smiles_col not in df.columns:
                raise ValueError(f"Missing column '{self.smiles_col}' in {self.input_path}")
            for _, row in df.iterrows():
                yield {k: str(v) for k, v in row.to_dict().items()}
            return

        if self.input_path.suffix.lower() in {".txt", ".smi"}:
            with self.input_path.open("r", encoding="utf-8") as f:
                for line in f:
                    smi = line.strip()
                    if not smi:
                        continue
                    yield {self.smiles_col: smi}
            return

        raise ValueError(f"Unsupported input type: {self.input_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TSV/CSV with SMILES column, or .txt/.smi with SMILES per line")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--cache-dir", default="./pubchem_cache")
    ap.add_argument("--no-pubchem-props", action="store_true")
    args = ap.parse_args()

    runner = StructureLookupRunner(
        input_path=args.input,
        outdir=args.outdir,
        smiles_col=args.smiles_col,
        cache_dir=args.cache_dir,
        use_pubchem_properties=not args.no_pubchem_props,
    )
    out_tsv, out_jsonl = runner.run()
    print("Wrote:", out_tsv)
    print("Wrote:", out_jsonl)


if __name__ == "__main__":
    main()