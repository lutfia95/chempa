from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class PubChemEvidence:
    found_in_pubchem: bool
    cid: Optional[int]
    pubmed_count: int
    patent_count: int
    synonym_sample: str


class PubChemClient:
    def __init__(
        self,
        cache_dir: str | Path,
        sleep_s: float = 0.2,
        timeout_s: float = 20.0,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_s = sleep_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def evidence_by_inchikey(self, inchikey: str) -> PubChemEvidence:
        if not inchikey:
            return PubChemEvidence(False, None, 0, 0, "")

        cid = self._cid_from_inchikey(inchikey)
        if cid is None:
            return PubChemEvidence(False, None, 0, 0, "")

        syns = self._synonyms(cid)
        pubmed = self._xrefs_count(cid, "PubMedID")
        patents = self._xrefs_count(cid, "PatentID")

        sample = "; ".join(syns[:5]) if syns else ""
        return PubChemEvidence(True, cid, pubmed, patents, sample)

    def properties_by_cid(self, cid: int) -> Dict[str, str]:
        """
        Pull a small, useful set of PubChem computed/recorded properties.
        Not all will be present for all CIDs.
        """
        props = [
            "MolecularWeight",
            "XLogP",
            "TPSA",
            "HBondDonorCount",
            "HBondAcceptorCount",
            "RotatableBondCount",
            "ExactMass",
            "FormalCharge",
            "IsotopeAtomCount",
        ]
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{','.join(props)}/JSON"

        data = self._get_json(url, cache_key=f"cid_{cid}_props.json")
        if not data:
            return {}

        out: Dict[str, str] = {}
        try:
            propset = data["PropertyTable"]["Properties"][0]
            for k, v in propset.items():
                if k == "CID":
                    continue
                out[f"pubchem_{k}"] = str(v)
        except Exception:
            return {}

        return out

    def _cid_from_inchikey(self, inchikey: str) -> Optional[int]:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
        data = self._get_json(url, cache_key=f"inchikey_{inchikey}_cids.json")
        if not data:
            return None
        try:
            cids = data["IdentifierList"]["CID"]
            return int(cids[0]) if cids else None
        except Exception:
            return None

    def _synonyms(self, cid: int) -> list[str]:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        data = self._get_json(url, cache_key=f"cid_{cid}_synonyms.json")
        if not data:
            return []
        try:
            syns = data["InformationList"]["Information"][0].get("Synonym", [])
            return [str(x) for x in syns]
        except Exception:
            return []

    def _xrefs_count(self, cid: int, xref_type: str) -> int:
        """
        xref_type examples: PubMedID, PatentID
        """
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/{xref_type}/JSON"
        data = self._get_json(url, cache_key=f"cid_{cid}_xrefs_{xref_type}.json")
        if not data:
            return 0
        try:
            info = data["InformationList"]["Information"][0]
            arr = info.get(xref_type, [])
            return len(arr) if isinstance(arr, list) else 0
        except Exception:
            return 0

    def _get_json(self, url: str, cache_key: str) -> Dict[str, Any] | None:
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.sleep_s)
                r = requests.get(url, timeout=self.timeout_s)
                if r.status_code == 200:
                    data = r.json()
                    cache_path.write_text(json.dumps(data), encoding="utf-8")
                    return data
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = str(e)
                time.sleep(0.5 * (attempt + 1))

        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inchikey", required=True)
    ap.add_argument("--cache-dir", default="./pubchem_cache")
    args = ap.parse_args()

    cli = PubChemClient(cache_dir=args.cache_dir)
    ev = cli.evidence_by_inchikey(args.inchikey)
    print(ev)


if __name__ == "__main__":
    main()