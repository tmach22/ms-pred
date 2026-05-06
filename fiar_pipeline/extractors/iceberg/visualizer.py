"""
ICEBERGDAGVisualizer
====================
Diagnostic tool for visualising the ICEBERG fragmentation DAG with actual
2D RDKit molecular structures rendered directly inside each node.

The separate fragment grid has been removed.  All information (structure,
formula, prob_gen, exact_mass, max_broken) now lives in the unified DAG view.

Requires: networkx, matplotlib, RDKit, Pillow — all present in ms-gen env.

Usage
-----
from fiar_pipeline.extractors.iceberg.visualizer import ICEBERGDAGVisualizer
from fiar_pipeline.extractors.iceberg import ICEBERGScalpel

scalpel = ICEBERGScalpel.from_config(cfg["iceberg"])
frag_hash_to_entry, fragments = scalpel.extract_with_dag(smiles, ce, mz, adduct)

viz = ICEBERGDAGVisualizer(output_dir="fiar_pipeline/results/dag_diagnostics")
dag_path = viz.export_all(smiles, frag_hash_to_entry, fragments, prefix="aspirin")
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — no display needed on compute nodes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


class ICEBERGDAGVisualizer:
    """
    Renders the ICEBERG fragmentation DAG with 2D molecular structures embedded
    directly inside the nodes.

    Layout: depth-stratified horizontal layers (root at top).
    Edges:  drawn behind node images via zorder management.
    Nodes:  RDKit MolDraw2DCairo PNG bytes decoded in-memory via PIL — no
            temporary files written.  Falls back to a formula text box if
            a SMILES is unavailable for a given node.
    """

    def __init__(self, output_dir: str = "fiar_pipeline/data/viz"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mol_to_rgba(
        self, smiles: Optional[str], mol_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Render a SMILES string to an RGBA numpy array entirely in memory.
        Returns None if SMILES is missing, invalid, or rendering fails.
        """
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(*mol_size)
            drawer.drawOptions().addStereoAnnotation = False
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
            return np.array(img)
        except Exception:
            return None

    def _build_frag_lookup(
        self, fragments
    ) -> Dict[Tuple[str, int, int], object]:
        """
        Build a (formula, max_broken, tree_depth) → FragmentRecord lookup.

        When multiple records share the same key (structural isomers at the same
        depth and bond-break count), the highest-prob_gen entry wins because
        fragments arrive pre-sorted descending.

        Storing the full record (rather than just smiles+mass) lets the renderer
        access root_mol and cleavage_atom_indices for highlighted rendering.
        """
        lookup: Dict[Tuple[str, int, int], object] = {}
        if not fragments:
            return lookup
        for rec in fragments:
            key = (rec.formula, rec.max_broken, rec.tree_depth)
            if key not in lookup:
                lookup[key] = rec
        return lookup

    def _mol_to_rgba_from_record(
        self, frag_rec, mol_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Render a fragment with cleavage-site highlighting entirely in memory.

        Strategy (Option C from the spec): use the root Mol and the fragment's
        atom index set from FragmentEngine.get_draw_dict() directly, so atom
        indices are never scrambled by a SMILES round-trip.

        Steps
        -----
        1. Build the submolecule by deleting non-fragment atoms from a RWMol
           copy of root_mol (reverse-order deletion preserves index arithmetic).
        2. Compute a fresh 2D layout for the submolecule.
        3. Map cleavage_atom_indices (root space) → submol space via a
           pre-computed offset table that is O(1) per atom.
        4. Draw with orange highlights on the attachment atoms.
        5. Decode the Cairo PNG bytes in-memory; no temp files written.

        Falls back to _mol_to_rgba(smiles) on any exception.
        """
        root_mol = getattr(frag_rec, "root_mol", None)
        frag_atom_indices = getattr(frag_rec, "frag_atom_indices", None)
        cleavage_atom_indices = getattr(frag_rec, "cleavage_atom_indices", None) or []

        if root_mol is None or not frag_atom_indices:
            return self._mol_to_rgba(frag_rec.smiles, mol_size)

        try:
            # ── Build submolecule ─────────────────────────────────────────────
            # Pre-compute root → sub index mapping BEFORE any removal:
            # after removing all non-fragment atoms in descending index order,
            # fragment atom `root_idx` lands at position equal to the number of
            # kept atoms with smaller root indices — i.e. its rank in sorted order.
            sorted_kept = sorted(frag_atom_indices)
            root_to_sub = {root_idx: sub_idx
                           for sub_idx, root_idx in enumerate(sorted_kept)}

            rw = Chem.RWMol(root_mol)
            for atom_idx in sorted(
                (i for i in range(root_mol.GetNumAtoms())
                 if i not in frag_atom_indices),
                reverse=True,
            ):
                rw.RemoveAtom(atom_idx)

            try:
                Chem.SanitizeMol(rw)
            except Exception:
                pass

            sub_mol = rw.GetMol()
            if AllChem.Compute2DCoords(sub_mol) != 0:
                return self._mol_to_rgba(frag_rec.smiles, mol_size)

            # ── Map cleavage atoms to submol indices ──────────────────────────
            sub_cleavage = [root_to_sub[i] for i in cleavage_atom_indices
                            if i in root_to_sub]

            # Orange (1.0, 0.5, 0.0) for attachment-point atoms.
            # Bonds entirely within the cleavage set are also highlighted so
            # ring-attachment sites read clearly (e.g. two adjacent ring atoms
            # both bonded to the removed group).
            cleavage_set = set(sub_cleavage)
            highlight_bonds = [
                b.GetIdx() for b in sub_mol.GetBonds()
                if b.GetBeginAtomIdx() in cleavage_set
                and b.GetEndAtomIdx() in cleavage_set
            ]
            atom_colors = {i: (1.0, 0.5, 0.0) for i in sub_cleavage}
            bond_colors = {i: (1.0, 0.5, 0.0) for i in highlight_bonds}

            # ── Render ────────────────────────────────────────────────────────
            drawer = rdMolDraw2D.MolDraw2DCairo(*mol_size)
            drawer.drawOptions().addStereoAnnotation = False
            drawer.DrawMolecule(
                sub_mol,
                highlightAtoms=sub_cleavage,
                highlightBonds=highlight_bonds,
                highlightAtomColors=atom_colors,
                highlightBondColors=bond_colors,
            )
            drawer.FinishDrawing()
            img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
            return np.array(img)

        except Exception:
            return self._mol_to_rgba(frag_rec.smiles, mol_size)

    # ── Main plot ─────────────────────────────────────────────────────────────

    def plot_dag(
        self,
        root_smiles: str,
        frag_hash_to_entry: dict,
        fragments=None,
        title: str = "Fragmentation DAG",
        save_name: Optional[str] = None,
        max_nodes: int = 80,
        mol_size: Tuple[int, int] = (160, 120),
        zoom: float = 0.45,
    ) -> Path:
        """
        Render a hierarchical DAG with 2D RDKit structures embedded in nodes.

        Parameters
        ----------
        root_smiles         : SMILES of the precursor (shown in title only)
        frag_hash_to_entry  : raw dict from model.predict_mol() / extract_with_dag()
        fragments           : List[FragmentRecord] from extract_with_dag();
                              provides SMILES + exact_mass for each node image
        max_nodes           : cap on how many DAG nodes to render
        mol_size            : (width_px, height_px) for each in-memory RDKit render
        zoom                : OffsetImage zoom factor (scales rendered px → display pts)
        """
        # ── Graph construction ────────────────────────────────────────────────
        G = nx.DiGraph()
        depth_map: Dict[str, int] = {}
        entries = list(frag_hash_to_entry.items())[:max_nodes]

        for frag_hash, entry in entries:
            G.add_node(frag_hash)
            depth_map[frag_hash] = entry.get("tree_depth", 0)

        for frag_hash, entry in entries:
            for parent_hash in entry.get("parents", []):
                if parent_hash in G and frag_hash in G:
                    G.add_edge(parent_hash, frag_hash,
                               broken=entry.get("max_broken", 0))

        max_depth = max(depth_map.values(), default=0)

        # ── Depth-stratified layout ───────────────────────────────────────────
        # Each depth occupies a horizontal row; nodes within a row are evenly
        # spread along x and centred at x=0.  Depth 0 (root) sits at y=0,
        # deeper layers step down by 1 unit each.
        pos: Dict[str, tuple] = {}
        for depth in range(max_depth + 1):
            layer = [n for n, d in depth_map.items() if d == depth]
            for i, node in enumerate(layer):
                pos[node] = (i - len(layer) / 2.0, -depth)

        # ── Figure sizing ─────────────────────────────────────────────────────
        widest = max(
            sum(1 for d in depth_map.values() if d == dep)
            for dep in range(max_depth + 1)
        )
        fig_w = max(30, widest * 2.8)
        fig_h = max(20, (max_depth + 1) * 5.0)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_title(
            f"{title}\nRoot: {root_smiles[:90]}", fontsize=10, pad=12
        )
        ax.axis("off")

        cmap = plt.cm.plasma

        # ── Edges — drawn first, then zorder forced to 1 on returned artists ──
        # nx.draw_networkx_edges does not accept a zorder kwarg in NetworkX 3.x;
        # we set it on the returned FancyArrowPatch / PathCollection instead.
        edge_artists = nx.draw_networkx_edges(
            G, pos, ax=ax,
            arrows=True, arrowsize=14,
            alpha=0.50, edge_color="#555555",
            connectionstyle="arc3,rad=0.08",
            min_source_margin=20, min_target_margin=20,
        )
        if edge_artists is not None:
            artists = edge_artists if hasattr(edge_artists, "__iter__") \
                else [edge_artists]
            for a in artists:
                if hasattr(a, "set_zorder"):
                    a.set_zorder(1)

        edge_label_texts = nx.draw_networkx_edge_labels(
            G, pos, ax=ax,
            edge_labels={(u, v): f"b{d['broken']}"
                         for u, v, d in G.edges(data=True)},
            font_size=5, alpha=0.70,
        )
        for txt in edge_label_texts.values():
            txt.set_zorder(1)

        # ── Node images — zorder=2 ────────────────────────────────────────────
        frag_lookup = self._build_frag_lookup(fragments)

        # Text is placed below each AnnotationBbox.  The image occupies
        # mol_size[1]*zoom display points vertically (centred on the node),
        # so the bottom edge sits mol_size[1]*zoom/2 pts below the anchor.
        text_pt_offset = mol_size[1] * zoom / 2 + 7

        for frag_hash, entry in frag_hash_to_entry.items():
            if frag_hash not in pos:
                continue

            depth    = depth_map.get(frag_hash, 0)
            formula  = entry.get("form", "")
            prob_gen = entry.get("prob_gen", 0.0)
            max_brok = entry.get("max_broken", 0)
            x, y     = pos[frag_hash]
            node_color = cmap(depth / max(max_depth, 1))

            frag_rec   = frag_lookup.get((formula, max_brok, depth))
            exact_mass = frag_rec.exact_mass if frag_rec is not None else None
            img_arr    = (self._mol_to_rgba_from_record(frag_rec, mol_size)
                          if frag_rec is not None
                          else None)

            if img_arr is not None:
                oi = OffsetImage(img_arr, zoom=zoom)
                oi.image.axes = ax
                ab = AnnotationBbox(
                    oi, (x, y),
                    frameon=True,
                    pad=0.05,
                    bboxprops=dict(
                        boxstyle="round,pad=0.1",
                        edgecolor=node_color,
                        linewidth=2.0,
                        facecolor="white",
                        alpha=0.95,
                    ),
                    zorder=2,
                )
                ax.add_artist(ab)
            else:
                # Fallback: formula text in a depth-coloured box
                ax.text(
                    x, y, formula or "?",
                    ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="lightyellow",
                              edgecolor=node_color,
                              linewidth=1.5, alpha=0.95),
                    zorder=2,
                )

            # Metadata strip — zorder=3 (above images)
            mass_str = f"{exact_mass:.3f}" if exact_mass is not None else "?"
            ax.annotate(
                f"p={prob_gen:.3f}  m={mass_str}  b{max_brok}",
                xy=(x, y),
                xytext=(0, -text_pt_offset),
                textcoords="offset points",
                ha="center", va="top",
                fontsize=5.5, color="#222222",
                zorder=3,
            )

        # ── Depth colorbar ────────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(0, max_depth)
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Tree Depth",
                     fraction=0.012, pad=0.01)

        save_name = save_name or f"{title.replace(' ', '_')[:40]}_dag.png"
        out = self.output_dir / save_name
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer] Embedded DAG → {out}")
        return out

    # ── One-shot entry point ──────────────────────────────────────────────────

    def export_all(
        self,
        root_smiles: str,
        frag_hash_to_entry: dict,
        fragments,
        prefix: str = "mol",
    ) -> Path:
        """Render the unified embedded-structure DAG and return its Path."""
        return self.plot_dag(
            root_smiles=root_smiles,
            frag_hash_to_entry=frag_hash_to_entry,
            fragments=fragments,
            title=f"{prefix} DAG",
            save_name=f"{prefix}_dag.png",
        )
