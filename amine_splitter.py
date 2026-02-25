from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem


PathLike = Union[str, Path]


@dataclass(frozen=True)
class AmineSplitStats:
    total_unique: int
    primary_aliphatic: int
    secondary_aliphatic: int
    aromatic_amines: int
    others: int


class AmineSplitter:
    """
    Split the unique list:
      - primary_aliphatic_amines.tsv
      - secondary_aliphatic_amines.tsv
      - aromatic_amines.tsv
      - others.tsv

    Definition used (pragmatic, RDKit-based):
      - "amine N" = nitrogen that is:
          * not aromatic itself
          * not amide / sulfonamide / carbamate / urea-like (i.e., N single-bonded to a carbonyl/sulfonyl carbon)
          * and has at least one H (primary/secondary)
      - "aromatic amine" = amine N bonded to an aromatic atom (aniline-like), e.g. Ar-NH2 or Ar-NHR
      - "aliphatic (alkyl) amine" = amine N with NO aromatic neighbors, and carbon neighbors are sp3 (best-effort)

    Primary vs Secondary (for the qualifying N):
      - primary: N has 1 carbon substituent and 2 H (total degree ~1, plus Hs)
      - secondary: N has 2 carbon substituents and 1 H

    Input file is NOT modified. Outputs go to output_dir.
    """

    def __init__(
        self,
        input_unique_tsv: PathLike,
        output_dir: PathLike,
        smiles_col: str = "SMILES",
    ) -> None:
        self.input_unique_tsv = Path(input_unique_tsv)
        self.output_dir = Path(output_dir)
        self.smiles_col = smiles_col

        self.output_dir.mkdir(parents=True, exist_ok=True)


    def split(self) -> AmineSplitStats:
        df = pd.read_csv(self.input_unique_tsv, sep="\t", dtype=str, keep_default_na=False)
        if self.smiles_col not in df.columns:
            raise ValueError(f"Missing column '{self.smiles_col}' in {self.input_unique_tsv}")

        buckets = {
            "primary_aliphatic": [],
            "secondary_aliphatic": [],
            "aromatic_amines": [],
            "others": [],
        }

        for i, row in df.iterrows():
            smi = (row.get(self.smiles_col) or "").strip()
            mol = Chem.MolFromSmiles(smi) if smi else None

            if mol is None:
                buckets["others"].append(row)
                continue

            cls = self._classify_molecule(mol)

            if cls == "primary_aliphatic":
                buckets["primary_aliphatic"].append(row)
            elif cls == "secondary_aliphatic":
                buckets["secondary_aliphatic"].append(row)
            elif cls == "aromatic_amine":
                buckets["aromatic_amines"].append(row)
            else:
                buckets["others"].append(row)

        self._write_bucket(df.columns.tolist(), buckets["primary_aliphatic"], self.output_dir / "primary_aliphatic_amines.tsv")
        self._write_bucket(df.columns.tolist(), buckets["secondary_aliphatic"], self.output_dir / "secondary_aliphatic_amines.tsv")
        self._write_bucket(df.columns.tolist(), buckets["aromatic_amines"], self.output_dir / "aromatic_amines.tsv")
        self._write_bucket(df.columns.tolist(), buckets["others"], self.output_dir / "others.tsv")

        total = len(df)
        p = len(buckets["primary_aliphatic"])
        s = len(buckets["secondary_aliphatic"])
        a = len(buckets["aromatic_amines"])
        o = len(buckets["others"])

        return AmineSplitStats(
            total_unique=total,
            primary_aliphatic=p,
            secondary_aliphatic=s,
            aromatic_amines=a,
            others=o,
        )

    def _classify_molecule(self, mol: Chem.Mol) -> str:
        """
        Return one of:
          - "primary_aliphatic"
          - "secondary_aliphatic"
          - "aromatic_amine"
          - "other"
        Priority: if any aromatic amine N exists -> aromatic_amine,
                  else if any primary/secondary aliphatic exists -> that (primary wins over secondary),
                  else other.
        """
        found_primary = False
        found_secondary = False
        found_aromatic_amine = False

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 7:
                continue

            if not self._is_amine_n(atom):
                continue

            if self._is_aromatic_amine_n(atom):
                found_aromatic_amine = True
                continue

            # aliphatic candidate
            kind = self._primary_or_secondary(atom)
            if kind == "primary":
                found_primary = True
            elif kind == "secondary":
                found_secondary = True

        if found_aromatic_amine:
            return "aromatic_amine"
        if found_primary:
            return "primary_aliphatic"
        if found_secondary:
            return "secondary_aliphatic"
        return "other"

    @staticmethod
    def _is_amine_n(n: Chem.Atom) -> bool:
        """
        Basic 'amine nitrogen' filter:
          - N, not aromatic itself
          - not positively charged quaternary-only (no H, 4 substituents)
          - exclude amide/sulfonamide/carbamate/urea-like via carbonyl/sulfonyl adjacency
          - exclude nitrile N (triple-bonded carbon)
        """
        if n.GetIsAromatic(): # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Atom.GetIsAromatic
            return False

        # exclude nitrile: C#N or N#C
        for b in n.GetBonds():
            if b.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                other = b.GetOtherAtom(n)
                if other.GetAtomicNum() == 6:
                    return False

        # exclude "amide-like": N single-bonded to carbonyl carbon (C(=O)-N),
        # sulfonamide: S(=O)(=O)-N, carbamate/urea are covered by carbonyl adjacency too.
        for nb in n.GetNeighbors(): # Returns a read-only sequence of the atom’s neighbors
            z = nb.GetAtomicNum() # Returns the atomic number.

            # carbonyl adjacency: N - C where that C has a double-bond O/S
            if z == 6 and _has_double_bond_to(nb, {8, 16}):
                return False

            # sulfonyl adjacency: N - S where S has double-bond O
            if z == 16 and _has_double_bond_to(nb, {8}):
                return False

        # require at least one H to be primary/secondary (not tertiary/quaternary-only)
        if n.GetTotalNumHs(includeNeighbors=True) <= 0:
            return False

        return True

    @staticmethod
    def _is_aromatic_amine_n(n: Chem.Atom) -> bool:
        """
        Aromatic amine: amine N bonded to at least one aromatic atom (aniline-like).
        """
        for nb in n.GetNeighbors():
            if nb.GetIsAromatic():
                return True
        return False

    @staticmethod
    def _primary_or_secondary(n: Chem.Atom) -> Optional[str]:
        """
        Classify amine N as primary or secondary based on:
          - number of carbon substituents (C neighbors)
          - number of attached hydrogens

        Only meaningful for aliphatic amines (no aromatic neighbors).
        """
        h = n.GetTotalNumHs(includeNeighbors=True)
        c_neighbors = sum(1 for nb in n.GetNeighbors() if nb.GetAtomicNum() == 6)

        # Best-effort "alkyl": carbon neighbors should be sp3 (avoid anilines already filtered),
        # but also avoid imines etc where carbon is sp2 with double bonds.
        # We'll require C neighbor is not aromatic and not carbonyl carbon.
        for nb in n.GetNeighbors():
            if nb.GetAtomicNum() != 6:
                continue
            if nb.GetIsAromatic():
                return None
            if _has_double_bond_to(nb, {8, 16}):
                return None

        if c_neighbors == 1 and h >= 2:
            return "primary"
        if c_neighbors == 2 and h >= 1:
            return "secondary"
        return None

    @staticmethod
    def _write_bucket(columns: List[str], rows: List[pd.Series], out_path: Path) -> None:
        out_df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
        out_df.to_csv(out_path, sep="\t", index=False)


def _has_double_bond_to(atom: Chem.Atom, target_atomic_nums: set[int]) -> bool:
    """Return True if atom has a double bond to any element in target_atomic_nums."""
    for b in atom.GetBonds():
        if b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            other = b.GetOtherAtom(atom)
            if other.GetAtomicNum() in target_atomic_nums:
                return True
    return False