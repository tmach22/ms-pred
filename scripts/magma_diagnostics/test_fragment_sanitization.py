import multiprocessing as mp
import time
import os
import sys
from rdkit import Chem

# Ensure the src directory is in the path so we can import ms-pred modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

try:
    from ms_pred.magma.fragmentation import FragmentEngine
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import FragmentEngine. {e}")
    sys.exit(1)

def test_magma_engine(mol, max_broken_bonds=2):
    try:
        engine = FragmentEngine(mol, max_broken_bonds=max_broken_bonds, max_tree_depth=2)
        engine.generate_fragments()
        if len(engine.frag_to_entry) > 1:
            return True, "Success"
        else:
            return False, "Engine ran but generated 0 sub-fragments."
    except Exception as e:
        return False, str(e)

def process_single_fragment(smiles):
    results = {
        "smiles": smiles,
        "standard_pass": False,
        "unsanitized_pass": False,
        "dummy_capped_pass": False,
        "radical_pass": False,
        "errors": {}
    }

    # 1. Standard
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            succ, msg = test_magma_engine(mol)
            results["standard_pass"] = succ
            if not succ: results["errors"]["standard"] = msg
    except Exception as e: results["errors"]["standard"] = f"RDKit parse failed: {e}"

    # 2. Unsanitized
    try:
        mol_uns = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol_uns:
            mol_uns.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol_uns)
            succ, msg = test_magma_engine(mol_uns)
            results["unsanitized_pass"] = succ
            if not succ: results["errors"]["unsanitized"] = msg
    except Exception as e: results["errors"]["unsanitized"] = f"Parse failed: {e}"

    # 3. Dummy Capping (*)
    try:
        capped = smiles.replace("[R]", "*")
        mol_cap = Chem.MolFromSmiles(capped)
        if mol_cap:
            succ, msg = test_magma_engine(mol_cap)
            results["dummy_capped_pass"] = succ
            if not succ: results["errors"]["dummy"] = msg
    except Exception as e: results["errors"]["dummy"] = f"Parse failed: {e}"

    # 4. Explicit Radicals
    try:
        mol_rad = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol_rad:
            mol_rad.UpdatePropertyCache(strict=False)
            for atom in mol_rad.GetAtoms():
                if atom.GetExplicitValence() < atom.GetTotalValence():
                    atom.SetNumRadicalElectrons(1)
            Chem.SanitizeMol(mol_rad)
            succ, msg = test_magma_engine(mol_rad)
            results["radical_pass"] = succ
            if not succ: results["errors"]["radical"] = msg
    except Exception as e: results["errors"]["radical"] = f"Parse failed: {e}"

    return results

def main():
    mock_fragments = [
        "c1ccccc1",                # Stable
        "[R]c1ccccc1",             # Phenyl radical
        "C[C](C)C",                # Tert-butyl radical
        "[R]C=CC=CC=C[R]",         # Opened benzene ring
        "O=C(O)C[R]"               # Acetic acid missing methyl
    ] * 100 # Total 500 tests

    num_workers = 16 # Safe limit for our 32-thread machine
    print(f"Running diagnostics on {len(mock_fragments)} fragments using {num_workers} workers...")

    start = time.time()
    with mp.Pool(processes=num_workers) as pool:
        batch_results = pool.map(process_single_fragment, mock_fragments)
    print(f"Completed in {time.time()-start:.2f} seconds.\n")

    total = len(mock_fragments)
    passes = {"Standard": 0, "Unsanitized": 0, "Dummy_Capped": 0, "Radical_Assigned": 0}

    print("--- CRASH LOGS (Sample) ---")
    error_samples = 0
    for r in batch_results:
        if r["standard_pass"]: passes["Standard"] += 1
        if r["unsanitized_pass"]: passes["Unsanitized"] += 1
        if r["dummy_capped_pass"]: passes["Dummy_Capped"] += 1
        if r["radical_pass"]: passes["Radical_Assigned"] += 1

        # Print a few errors for the Unsanitized approach to check InChI failures
        if not r["unsanitized_pass"] and "unsanitized" in r["errors"] and error_samples < 5:
            if r["smiles"] != "c1ccccc1": # ignore the control
                print(f"[Unsanitized Crash] {r['smiles']}: {r['errors']['unsanitized']}")
                error_samples += 1

    print("\n========================================")
    print("      FINAL EXECUTION SUMMARY           ")
    print("========================================")
    for method, count in passes.items():
        print(f"{method:18}: {count}/{total} ({(count/total)*100:.1f}%)")
    print("========================================")

if __name__ == '__main__':
    main()
