from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from io import BytesIO
try:
    from rdkit import Checm
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False


PathLike = Union[str, Path]

"""
def dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False,
              unsafe_hash=False, frozen=False, match_args=True,
              kw_only=False, slots=False, weakref_slot=False):
    Add dunder methods based on the fields defined in the class.

    Examines PEP 526 __annotations__ to determine fields.

    If init is true, an __init__() method is added to the class. If repr
    is true, a __repr__() method is added. If order is true, rich
    comparison dunder methods are added. If unsafe_hash is true, a
    __hash__() method is added. If frozen is true, fields may not be
    assigned to after instance creation. If match_args is true, the
    __match_args__ tuple is added. If kw_only is true, then by default
    all fields are keyword-only. If slots is true, a new class with a
    __slots__ attribute is returned.
 
"""
@dataclass(init=True, repr=True, eq=True, frozen = True)
class UniqueSmilesSummary:
    total_rows: int
    unique_smiles: int
    duplicate_rows: int

class SDFTSVAnalyzer:
    def __init__(self,
                 tsv_path: PathLike, 
                 smiles_col = "SMILES",
                 keep: str = "first"
                 ) -> None:
        self.tsv_path = Path(tsv_path)
        self.smiles_col = smiles_col
        self.keep = keep
        self.df: Optional(pd.DataFrame) = None
        
    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.tsv_path, sep="\t", dtype=str, 
                         keep_default_na=False)
        if self.smiles_col not in df.columns:
            raise ValueError(f"Missing column {self.smiles_col} in {self.tsv_path}")
        self.df = df 
        return df
    
    def unique_smiles_summary(self) -> UniqueSmilesSummary:
        df = self._require_df()
        total = len(df)
        uniq = df[self.smiles_col].nunique(dropna=True)
        dup = total - uniq
        return UniqueSmilesSummary(total_rows=total, unique_smiles=uniq, duplicate_rows=dup)
    
    def write_unique_tsv(
        self, 
        out_tsv: PathLike,
        subset_cols: Optional[Sequence[str]] = None
    ) -> Path:
        df = self._require_df()

        out_path = Path(out_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if subset_cols is None:
            subset_cols = list(df.columns)

        missing = [c for c in subset_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested columns not in TSV: {missing}")

        ddu = (
            df.drop_duplicates(subset=[self.smiles_col], keep=self.keep)
              .loc[:, list(subset_cols)]
        )
        ddu.to_csv(out_path, sep="\t", index=False)
        return out_path
    

    @staticmethod
    def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")
    
    def plot_numeric_distributions(
        self,
        columns: Optional[Sequence[str]] = None,
        bins: int = 50,
        use_unique: bool = True,
        out_dir: Optional[PathLike] = None,
        show: bool = True,
    ) -> List[Path]:
        df = self._require_df()

        if use_unique:
            df = df.drop_duplicates(subset=[self.smiles_col], keep=self.keep)

        if columns is None:
            candidates = [c for c in df.columns if c != self.smiles_col]
            cols: List[str] = []
            for c in candidates:
                s = self._to_numeric_series(df, c)
                if s.notna().mean() >= 0.5:
                    cols.append(c)
            columns = cols

        saved: List[Path] = []
        save_dir = Path(out_dir) if out_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        for c in columns:
            if c not in df.columns:
                continue
            s = self._to_numeric_series(df, c).dropna()
            if s.empty:
                continue

            plt.figure()
            plt.hist(s.values, bins=bins)
            plt.title(c)
            plt.xlabel(c)
            plt.ylabel("Count")

            if save_dir is not None:
                p = save_dir / f"{c}.png"
                plt.savefig(p, dpi=200, bbox_inches="tight")
                saved.append(p)

            if show:
                plt.show()
            else:
                plt.close()

        return saved
    
    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            return self.read()
        return self.df
    
