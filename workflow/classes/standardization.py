from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import pandas as pd

from workflow.classes.data_models import StandardizationSummary
from workflow.config import PipelineConfig


class RDKitStandardizer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardizationSummary]:
        Chem, rdMolDescriptors, Crippen, Lipinski, MurckoScaffold, rdMolStandardize = self._imports()

        smiles_column = self.config.input.smiles_column
        parent_strategy = self.config.standardization.parent_strategy.strip().lower()

        records: List[Dict[str, Any]] = []
        invalid_rows = 0
        duplicate_rows_removed = 0

        seen: Counter[str] = Counter()
        for row in df.to_dict(orient="records"):
            raw_smiles = str(row.get(smiles_column, "") or "").strip()
            record: Dict[str, Any] = dict(row)
            record["raw_smiles"] = raw_smiles
            record["is_valid"] = "0"
            record["invalid_reason"] = ""
            record["standardized_smiles"] = ""
            record["canonical_smiles"] = ""
            record["murcko_scaffold"] = ""
            record["is_duplicate"] = "0"
            record["duplicate_rank"] = "0"

            if not raw_smiles:
                record["invalid_reason"] = "empty_smiles"
                invalid_rows += 1
                records.append(record)
                continue

            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None:
                record["invalid_reason"] = "rdkit_parse_failed"
                invalid_rows += 1
                records.append(record)
                continue

            try:
                mol = self._apply_parent_strategy(rdMolStandardize, mol, parent_strategy)
                mol = self._normalize_prepared_mol(Chem, mol)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception as exc:
                record["invalid_reason"] = f"standardization_failed:{exc.__class__.__name__}"
                invalid_rows += 1
                records.append(record)
                continue

            record["is_valid"] = "1"
            record["canonical_smiles"] = canonical_smiles
            record["standardized_smiles"] = canonical_smiles
            record["murcko_scaffold"] = scaffold
            record["MW"] = f"{rdMolDescriptors.CalcExactMolWt(mol):.4f}"
            record["logP"] = f"{Crippen.MolLogP(mol):.4f}"
            record["HBD"] = str(Lipinski.NumHDonors(mol))
            record["HBA"] = str(Lipinski.NumHAcceptors(mol))
            record["TPSA"] = f"{rdMolDescriptors.CalcTPSA(mol):.4f}"
            record["RB"] = str(Lipinski.NumRotatableBonds(mol))
            record["Rings"] = str(rdMolDescriptors.CalcNumRings(mol))
            record["HeavyAtoms"] = str(mol.GetNumHeavyAtoms())
            record["NumAtoms"] = str(mol.GetNumAtoms())

            seen[canonical_smiles] += 1
            record["duplicate_rank"] = str(seen[canonical_smiles])
            if seen[canonical_smiles] > 1:
                record["is_duplicate"] = "1"
                duplicate_rows_removed += 1

            records.append(record)

        standardized = pd.DataFrame(records)
        if self.config.standardization.drop_invalid:
            standardized = standardized[standardized["is_valid"] == "1"].copy()
        if self.config.standardization.deduplicate:
            standardized = standardized[standardized["is_duplicate"] == "0"].copy()

        standardized.reset_index(drop=True, inplace=True)
        summary = StandardizationSummary(
            total_rows=len(df),
            invalid_rows=invalid_rows,
            duplicate_rows_removed=duplicate_rows_removed,
            valid_rows=len(standardized),
            parent_strategy=parent_strategy,
        )
        standardized.attrs["standardization_summary"] = asdict(summary)
        return standardized, summary

    def _normalize_prepared_mol(self, Chem: Any, mol: Any) -> Any:
        standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
        reparsed = Chem.MolFromSmiles(standardized_smiles)
        if reparsed is None:
            raise ValueError("reparse_after_standardization_failed")

        reparsed.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(reparsed)
        return reparsed

    def _apply_parent_strategy(self, rdMolStandardize: Any, mol: Any, strategy: str) -> Any:
        if strategy == "none":
            return mol
        if strategy == "fragment":
            return rdMolStandardize.FragmentParent(mol)
        if strategy == "charge":
            return rdMolStandardize.ChargeParent(mol)
        if strategy == "tautomer":
            return rdMolStandardize.TautomerParent(mol)
        if strategy == "super":
            return rdMolStandardize.SuperParent(mol)
        raise ValueError(
            "Unsupported parent strategy. Expected one of: none, fragment, charge, tautomer, super."
        )

    def _imports(self) -> Tuple[Any, Any, Any, Any, Any, Any]:
        try:
            from rdkit import Chem
            from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors
            from rdkit.Chem.Scaffolds import MurckoScaffold
            from rdkit.Chem.MolStandardize import rdMolStandardize
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RDKit is required for the standardization stage. "
                "Install RDKit in the runtime environment before running the workflow."
            ) from exc

        return Chem, rdMolDescriptors, Crippen, Lipinski, MurckoScaffold, rdMolStandardize
