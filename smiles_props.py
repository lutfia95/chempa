from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


@dataclass(frozen=True)
class SmilesParseResult:
    ok: bool
    canonical_smiles: str
    inchi: str
    inchikey: str
    error: str


class SmilesPropertyCalculator:
    def parse(self, smiles: str) -> SmilesParseResult:
        smi = (smiles or "").strip()
        if not smi:
            return SmilesParseResult(False, "", "", "", "empty_smiles")

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return SmilesParseResult(False, "", "", "", "rdkit_parse_failed")

        canonical = Chem.MolToSmiles(mol, canonical=True)

        inchi = ""
        inchikey = ""
        try:
            inchi = Chem.inchi.MolToInchi(mol)  # type: ignore[attr-defined]
            inchikey = Chem.inchi.MolToInchiKey(mol)  # type: ignore[attr-defined]
        except Exception:
            pass

        return SmilesParseResult(True, canonical, inchi, inchikey, "")

    def rdkit_props(self, smiles: str) -> Dict[str, str]:
        mol = Chem.MolFromSmiles((smiles or "").strip())
        if mol is None:
            return {}

        out: Dict[str, str] = {}
        out["MW"] = f"{Descriptors.MolWt(mol):.2f}"
        out["logP"] = f"{Crippen.MolLogP(mol):.2f}"
        out["HBD"] = str(Lipinski.NumHDonors(mol))
        out["HBA"] = str(Lipinski.NumHAcceptors(mol))
        out["TPSA"] = f"{rdMolDescriptors.CalcTPSA(mol):.2f}"
        out["RB"] = str(Lipinski.NumRotatableBonds(mol))
        out["Rings"] = str(rdMolDescriptors.CalcNumRings(mol))
        out["HeavyAtoms"] = str(mol.GetNumHeavyAtoms())
        out["NumAtoms"] = str(mol.GetNumAtoms())
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    args = ap.parse_args()

    c = SmilesPropertyCalculator()
    p = c.parse(args.smiles)
    print(p)
    print(c.rdkit_props(args.smiles))


if __name__ == "__main__":
    main()