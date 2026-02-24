from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase.pdfmetrics import stringWidth

    _HAVE_REPORTLAB = True
except Exception:
    _HAVE_REPORTLAB = False

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


PathLike = Union[str, Path]


@dataclass(frozen=True)
class MolRecord:
    """A single molecule record extracted from an SDF."""
    file: str
    index: int
    label: str
    smiles: str
    num_atoms: int
    props: Dict[str, str]


class SDFStructureReporter:
    """
    Read SDF files from a path (single SDF or directory), summarize molecules,
    compute sensible RDKit properties, and export TSV and/or PDF.

    Ussage in a notebook:
        from rdkit_sdf_reporter import SDFStructureReporter

        reporter = SDFStructureReporter(
            input_path="Structures of compounds_Xiaoyi_2026.02.12",
            label_prop=None,
            props=["MW", "logP", "HBD", "HBA", "TPSA", "RB", "Rings", "HeavyAtoms"]
        )

        summary = reporter.summarize()
        print(summary["files"], summary["molecules_total"])

        reporter.write_tsv("out/report.tsv")
        reporter.write_pdf("out/report.pdf")   # optional
    """

    def __init__(
        self,
        input_path: PathLike,
        label_prop: Optional[str] = None,
        props: Optional[Sequence[str]] = None,
        sanitize: bool = True,
        add_hs_for_depiction: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        self.label_prop = label_prop
        self.props = list(props) if props is not None else [
            "MW", "logP", "HBD", "HBA", "TPSA", "RB", "Rings", "HeavyAtoms"
        ]
        self.sanitize = sanitize
        self.add_hs_for_depiction = add_hs_for_depiction


    @staticmethod
    def get_number_of_atoms(mol: Chem.Mol) -> Optional[int]:
        """Return number of atoms (None if mol is None)."""
        if mol is None:
            return None
        return mol.GetNumAtoms()

    @staticmethod
    def pick_label(
        mol: Chem.Mol,
        label_prop: Optional[str],
        fallback_index: Optional[int],
    ) -> Optional[str]:
        """
        Choose a human-friendly label for a molecule.

        Priority:
          1) Provided SDF property name (label_prop)
          2) RDKit SDF title line (_Name)
          3) Common-ish IDs
          4) Fallback to "#<index>"
        """
        if mol is None:
            return None

        if label_prop and mol.HasProp(label_prop):
            v = mol.GetProp(label_prop).strip()
            if v:
                return v

        # RDKit stores SDF title line in _Name, source: https://github.com/rdkit/rdkit-orig/blob/master/rdkit/Chem/PropertyMol.py
        if mol.HasProp("_Name"):
            v = mol.GetProp("_Name").strip()
            if v:
                return v

        for k in ("ID", "Name", "MolID", "Compound_ID", "CompoundID", "title"):
            if mol.HasProp(k):
                v = mol.GetProp(k).strip()
                if v:
                    return v

        return f"#{fallback_index}" if fallback_index is not None else None

    @staticmethod
    def calc_builtin_props(mol: Chem.Mol) -> Dict[str, str]:
        """
        Compute a set of sane, widely-used properties.
        Failures are ignored (property not included).
        """
        out: Dict[str, str] = {}
        if mol is None:
            return out

        # MW, logP
        try:
            out["MW"] = f"{Descriptors.MolWt(mol):.2f}" # the exact molecular weight of the molecule, https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html
        except Exception:
            pass
        try:
            out["logP"] = f"{Crippen.MolLogP(mol):.2f}" # Wildman-Crippen LogP value
            # atomic-contribution method used to estimate the octanol/water partition coefficient
        except Exception:
            pass

        # MedChem-ish basics
        try:
            out["HBD"] = str(Lipinski.NumHDonors(mol)) # Number of Hydrogen Bond Donors
        except Exception:
            pass
        try:
            out["HBA"] = str(Lipinski.NumHAcceptors(mol)) # Number of Hydrogen Bond Acceptors
        except Exception:
            pass
        # ToDo: add, rdkit.Chem.Lipinski.NumHeteroatoms(x), rdkit.Chem.Lipinski.NumHeterocycles, 
        # rdkit.Chem.Lipinski.NumSaturatedCarbocycles
        # rdkit.Chem.Lipinski.NumSaturatedHeterocycles, rdkit.Chem.Lipinski.NumSaturatedRings, 
        # rdkit.Chem.Lipinski.NumSpiroAtoms, dkit.Chem.Lipinski.NumUnspecifiedAtomStereoCenters
        # rdkit.Chem.Lipinski.Phi
        try:
            out["TPSA"] = f"{rdMolDescriptors.CalcTPSA(mol):.2f}" # The Topological Polar Surface Area (TPSA) 
        except Exception:
            pass
        try:
            out["RB"] = str(Lipinski.NumRotatableBonds(mol)) # Simple rotatable bond definition. strict = NumRotatableBondsOptions.Strict - (default) does not count things like amide or ester bonds
        except Exception:
            pass
        try:
            out["Rings"] = str(rdMolDescriptors.CalcNumRings(mol)) # returns the number of rings for a molecule (https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.CalcNumRings)
        except Exception:
            pass
        try:
            out["HeavyAtoms"] = str(mol.GetNumHeavyAtoms()) # Returns the number of heavy atoms (atomic number >1) in the molecule.
        except Exception:
            pass

        return out

    @classmethod
    def get_prop_string(cls, mol: Chem.Mol, props: Sequence[str]) -> str:
        """
        Build a single-line string of properties.

        props can contain:
          - SDF property names (existing on molecule)
          - Built-ins: MW, logP, HBD, HBA, TPSA, RB, Rings, HeavyAtoms
        """
        built = cls.calc_builtin_props(mol)
        parts: List[str] = []

        for p in props:
            p = p.strip()
            if not p:
                continue

            if p in built:
                parts.append(f"{p}={built[p]}")
            elif mol is not None and mol.HasProp(p):
                v = mol.GetProp(p).strip().replace("\n", " ")
                if v:
                    parts.append(f"{p}={v}")

        return "  ".join(parts)

    @staticmethod
    def _safe_smiles(mol: Chem.Mol) -> str:
        """Return canonical SMILES or empty string on failure."""
        if mol is None:
            return ""
        try:
            return Chem.MolToSmiles(mol, canonical=True) # Returns the canonical SMILES string for a molecule
        except Exception:
            return ""


    def iter_sdf_files(self) -> Iterator[Path]:
        """Yield SDF files from input_path (file or directory), sorted."""
        if self.input_path.is_file():
            if self.input_path.suffix.lower() == ".sdf":
                yield self.input_path
            else:
                raise ValueError(f"Input file is not .sdf: {self.input_path}")
            return

        if self.input_path.is_dir():
            for p in sorted(self.input_path.rglob("*.sdf")):
                yield p
            return

        raise FileNotFoundError(f"Path not found: {self.input_path}")

    def _supplier(self, sdf_path: Path) -> Chem.SDMolSupplier:
        """
        Create an SDMolSupplier. sanitize controls RDKit sanitization.
        If sanitize=False, you can still compute some things, but many descriptors may fail.
        """
        return Chem.SDMolSupplier(str(sdf_path), sanitize=self.sanitize, removeHs=False)

    def iter_records(self) -> Iterator[MolRecord]:
        """
        Stream all molecules across all SDF files as MolRecord objects.

        Notes:
          - Invalid molecules (None) are skipped.
          - index is 1-based per file (for human-friendly labeling).
        """
        for sdf in self.iter_sdf_files():
            supplier = self._supplier(sdf)
            idx = 0
            for mol in supplier:
                idx += 1
                if mol is None:
                    continue

                label = self.pick_label(mol, self.label_prop, idx) or f"#{idx}"
                smiles = self._safe_smiles(mol)
                num_atoms = mol.GetNumAtoms()

                built = self.calc_builtin_props(mol)
                extra: Dict[str, str] = {}

                # Also pass through requested SDF properties (if any)
                for p in self.props:
                    p = p.strip()
                    if not p:
                        continue
                    if p in built:
                        continue
                    if mol.HasProp(p):
                        v = mol.GetProp(p).strip().replace("\n", " ")
                        if v:
                            extra[p] = v

                merged = dict(built)
                merged.update(extra)

                yield MolRecord(
                    file=str(sdf.name),
                    index=idx,
                    label=label,
                    smiles=smiles,
                    num_atoms=num_atoms,
                    props=merged,
                )

    def summarize(self) -> Dict[str, int]:
        """
        Return counts:
          - files: number of SDF files discovered
          - molecules_total: number of valid molecules across them
        """
        files = list(self.iter_sdf_files())
        mols = 0
        for sdf in files:
            supplier = self._supplier(sdf)
            for mol in supplier:
                if mol is not None:
                    mols += 1
        return {"files": len(files), "molecules_total": mols}

    def print_file_molecule_counts(self) -> None:
        """Print number of molecules per SDF file."""
        for sdf in self.iter_sdf_files():
            supplier = self._supplier(sdf)
            n = sum(1 for m in supplier if m is not None)
            print(f"[INFO] Number of molecules in {sdf}: {n}")


    def _tsv_header(self) -> List[str]:
        """
        TSV columns:
          SMILES first, then label/file/index/num_atoms, then requested props (builtins + SDF props).
        """
        builtin_order = ["MW", "logP", "HBD", "HBA", "TPSA", "RB", "Rings", "HeavyAtoms"]
        builtins = [p for p in self.props if p in builtin_order]
        builtins = [p for p in builtin_order if p in builtins]
        others = [p for p in self.props if p not in builtin_order]
        return ["SMILES", "Label", "File", "Index", "NumAtoms"] + builtins + others

    def write_tsv(self, out_tsv: PathLike) -> Path:
        """
        Write a TSV report with:
          - first column: SMILES
          - then metadata and properties

        This is the robust default output (no external services, no auth).
        """
        out_path = Path(out_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = self._tsv_header()
        cols_props = header[5:]  # property columns

        with out_path.open("w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")
            for r in self.iter_records():
                row: List[str] = [
                    r.smiles,
                    r.label,
                    r.file,
                    str(r.index),
                    str(r.num_atoms),
                ]
                for p in cols_props:
                    row.append(r.props.get(p, ""))
                f.write("\t".join(row) + "\n")

        return out_path


    @staticmethod
    def _mol_to_png_bytes(
        mol: Chem.Mol,
        size: Tuple[int, int] = (240, 180),
        kekulize: bool = True,
        add_hs: bool = False,
        legend: str = "",
    ) -> bytes:
        """
        Render molecule to PNG bytes using RDKit's 2D drawer.
        Uses PIL only if needed by downstream consumers; returns raw PNG bytes.
        """
        if mol is None:
            return b""

        m = Chem.Mol(mol)
        if add_hs:
            try:
                m = Chem.AddHs(m)
            except Exception:
                pass

        try:
            if m.GetNumConformers() == 0:
                AllChem.Compute2DCoords(m)
        except Exception:
            pass

        w, h = size
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        try:
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, m, legend=legend, kekulize=kekulize)
        except Exception:
            # fallback without kekulize
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, m, legend=legend, kekulize=False)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def write_pdf(
        self,
        out_pdf: PathLike,
        per_page: int = 6,
        image_size_mm: Tuple[float, float] = (65.0, 45.0),
        font_size: int = 9,
    ) -> Path:
        """
        Create a simple PDF report:
          - Each record shows a structure image + a compact property line.
          - No login/auth needed. Requires reportlab and (optionally) pillow.

        If reportlab isn't installed, raises RuntimeError. If pillow isn't installed,
        reportlab can still embed PNG via ImageReader in most cases, but pillow helps.

        Layout:
          - per_page entries, stacked top-to-bottom.
          - Each entry: image left, text right.
        """
        if not _HAVE_REPORTLAB:
            raise RuntimeError("PDF export requires reportlab. Install it or use write_tsv().")

        out_path = Path(out_pdf)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        c = canvas.Canvas(str(out_path), pagesize=A4)
        page_w, page_h = A4

        margin_x = 12 * mm
        margin_y = 12 * mm
        line_gap = 3 * mm

        img_w = image_size_mm[0] * mm
        img_h = image_size_mm[1] * mm

        # Compute row height so we can pack per_page entries reliably
        usable_h = page_h - 2 * margin_y
        row_h = usable_h / per_page
        
        def wrap_by_width(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
            """Word-wrap a string based on rendered width in points."""
            words = text.split()
            if not words:
                return [""]

            lines: List[str] = []
            cur = words[0]
            for w in words[1:]:
                trial = cur + " " + w
                if stringWidth(trial, font_name, font_size) <= max_width:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = w
            lines.append(cur)
            return lines

        def draw_entry(y_top: float, mol: Chem.Mol, title: str, text: str) -> None:
            # Render mol to PNG and draw it
            png = self._mol_to_png_bytes(
                mol,
                size=(int(img_w), int(img_h)),
                add_hs=self.add_hs_for_depiction,
                legend="",  # keep legend out of the image; print text instead
            )
            if png:
                img = ImageReader(BytesIO(png))
                c.drawImage(img, margin_x, y_top - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")

            # Text block to the right
            tx = margin_x + img_w + 8 * mm
            ty = y_top - 2 * mm
            c.setFont("Helvetica-Bold", font_size + 1)
            c.drawString(tx, ty, title)

            c.setFont("Helvetica", font_size)

            # Available width for text block (from tx to right margin)
            max_w = (page_w - margin_x) - tx

            # Wrap using real rendered width (no clipping)
            lines = wrap_by_width(text, "Helvetica", font_size, max_w)

            t_y = ty - (font_size + 2)
            max_lines = 8  # increase if you want more lines per entry
            for ln in lines[:max_lines]:
                c.drawString(tx, t_y, ln)
                t_y -= (font_size + 2)

        # We need access to the actual mol objects for depiction.
        # Stream them by re-reading each SDF supplier in parallel with records.
        for sdf in self.iter_sdf_files():
            supplier = self._supplier(sdf)
            idx = 0
            for mol in supplier:
                idx += 1
                if mol is None:
                    continue

                label = self.pick_label(mol, self.label_prop, idx) or f"#{idx}"
                smiles = self._safe_smiles(mol)
                num_atoms = mol.GetNumAtoms()
                prop_line = self.get_prop_string(mol, self.props)

                #title = f"{label}   ({sdf.name}  idx={idx}  atoms={num_atoms})"
                title = f"{label}   |  IDX: {idx}  |  Atoms: {num_atoms} | \nFile: {sdf.name} "
                text = f"SMILES={smiles}"
                if prop_line:
                    text += f"   {prop_line}"

                # Compute position within page
                # Entry number within current page (0..per_page-1)
                # We track global count per file (fine)
                entry_i = (idx - 1) % per_page
                y_top = page_h - margin_y - entry_i * row_h

                draw_entry(y_top, mol, title, text)

                # separator line
                y_sep = y_top - row_h + line_gap
                c.setLineWidth(0.5)
                c.line(margin_x, y_sep, page_w - margin_x, y_sep)

                if entry_i == per_page - 1:
                    c.showPage()

        c.save()
        return out_path