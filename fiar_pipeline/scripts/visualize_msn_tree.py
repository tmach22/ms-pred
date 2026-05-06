#!/usr/bin/env python3
"""
MSnLib ASCII Tree Visualizer
Reconstructs the hierarchical MS1 -> MS2 -> MS3 tree from flat NDJSON files.
"""

import sys
import json
from collections import defaultdict

def visualize_tree(json_path, max_molecules=2):
    # Data structure: MS1_mz -> {smiles: str, ms2_nodes: {ms2_mz -> {ms3_mz -> top_peaks}}}
    trees = {}
    
    print(f"[*] Scanning {json_path} for hierarchical trees...\n")
    
    with open(json_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
                
            precursor_mz = data.get('precursor_mz')
            msn_chain = data.get('msn_precursor_mzs', [])
            
            # We only care about full MS3 chains for this visualization
            if not precursor_mz or len(msn_chain) < 2:
                continue
                
            smiles = data.get('smiles', 'Unknown')
            ms2_mz = msn_chain[0]
            ms3_mz = msn_chain[1]
            
            # Grab actual spectrum peaks and sort by intensity to get the top 3
            peaks = data.get('mz', [])
            intensities = data.get('intensity', [])
            top_peaks = sorted(zip(peaks, intensities), key=lambda x: x[1], reverse=True)[:3]
            
            # Build the nested dictionary tree
            if precursor_mz not in trees:
                if len(trees) >= max_molecules:
                    continue # Wait until we finish populating the ones we have
                trees[precursor_mz] = {'smiles': smiles, 'ms2_nodes': defaultdict(dict)}
                
            trees[precursor_mz]['ms2_nodes'][ms2_mz][ms3_mz] = top_peaks

    # Print the ASCII Tree
    if not trees:
        print("[!] No MS3 trees found in the scanned lines.")
        return

    for ms1, info in trees.items():
        print(f"🟩 [MS1 Root] Precursor: {ms1:.4f} Da")
        print(f"   SMILES: {info['smiles']}")
        
        ms2_items = list(info['ms2_nodes'].items())
        for i, (ms2, ms3_dict) in enumerate(ms2_items):
            ms2_connector = "└──" if i == len(ms2_items) - 1 else "├──"
            ms2_pipe      = " "   if i == len(ms2_items) - 1 else "│"
            
            print(f"   {ms2_connector} 🟦 [MS2 Branch] Isolated Fragment: {ms2:.4f} Da")
            
            ms3_items = list(ms3_dict.items())
            for j, (ms3, peaks) in enumerate(ms3_items):
                ms3_connector = "└──" if j == len(ms3_items) - 1 else "├──"
                ms3_pipe      = " "   if j == len(ms3_items) - 1 else "│"
                
                print(f"   {ms2_pipe}    {ms3_connector} 🟪 [MS3 Branch] Isolated Sub-Fragment: {ms3:.4f} Da")
                
                for k, (mz, inty) in enumerate(peaks):
                    peak_connector = "└──" if k == len(peaks) - 1 else "├──"
                    print(f"   {ms2_pipe}    {ms3_pipe}    {peak_connector} 🔴 [Peak] {mz:.4f} Da (Intensity: {inty:,.0f})")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_msn_tree.py <path_to_json>")
        sys.exit(1)
    
    # Optional: pass max molecules as second arg
    max_mols = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    visualize_tree(sys.argv[1], max_mols)