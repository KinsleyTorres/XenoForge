from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
from openmm.app import PDBFile, Modeller, ForceField, Simulation, NoCutoff, HBonds
from openmm import LangevinIntegrator, unit, Platform
from pdbfixer import PDBFixer

# -----------------------------
# USER INPUTS
# -----------------------------
SMILES = "NC(N)=O"
ENZYME_PDB = "prediction.pdb"
OUTPUT_LIGAND = "ligand.pdb"

# -----------------------------
# 1 ▸ Generate ligand
# -----------------------------
mol = Chem.MolFromSmiles(SMILES)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)
rdmolfiles.MolToPDBFile(mol, OUTPUT_LIGAND)
print(f"[✓] Generated ligand PDB → {OUTPUT_LIGAND}")

# Get ligand energy
mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
mmff_ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
E_ligand = mmff_ff.CalcEnergy() * 4.184  # Convert to kJ/mol
print(f"[✓] Ligand energy: {E_ligand:.2f} kJ/mol")

# -----------------------------
# 2 ▸ Fix enzyme with terminal caps
# -----------------------------
print("\n[...] Cleaning AlphaFold structure...")
fixer = PDBFixer(filename=ENZYME_PDB)

# For AlphaFold structures
print("  - Removing water and heterogens...")
fixer.removeHeterogens(keepWater=False)

print("  - Finding missing residues (including terminal caps)...")
fixer.findMissingResidues()

# Check what's missing
if fixer.missingResidues:
    print(f"    Found missing residues at {len(fixer.missingResidues)} positions")
    for key, residues in fixer.missingResidues.items():
        if residues:
            print(f"      Chain {key[0]}, position {key[1]}: {residues}")

print("  - Finding missing atoms...")
fixer.findMissingAtoms()
if fixer.missingAtoms:
    print(f"    Found {len(fixer.missingAtoms)} residues with missing atoms")

print("  - Adding missing atoms and terminal caps...")
fixer.addMissingAtoms()

print("  - Adding hydrogens at pH 7.0...")
fixer.addMissingHydrogens(7.0)

# Save
with open("enzyme_cleaned.pdb", "w") as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)
print("[✓] Saved cleaned enzyme → enzyme_cleaned.pdb")

# -----------------------------
# 3 ▸ Energy minimization
# -----------------------------
try:
    print("\n[...] Computing enzyme energy with minimization...")
    forcefield = ForceField("amber14-all.xml")

    pdb = PDBFile("enzyme_cleaned.pdb")

    # Create system
    print("  - Creating force field system...")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        rigidWater=False
    )

    # Use CPU platform
    platform = Platform.getPlatformByName('CPU')
    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)

    sim = Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    # Check initial energy
    print("  - Checking initial energy...")
    state = sim.context.getState(getEnergy=True)
    E_initial = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"    Initial energy: {E_initial:.2e} kJ/mol")

    if E_initial > 1e6:  # More than 1 million kJ/mol
        print("\n  ⚠ Energy is very high - performing staged minimization...")

        # Stage 1: Resolve severe clashes
        print("\n  [Stage 1/3] Resolving severe clashes...")
        sim.minimizeEnergy(tolerance=1000 * unit.kilojoule / (unit.mole * unit.nanometer), maxIterations=2000)

        state = sim.context.getState(getEnergy=True)
        E_stage1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"    Energy: {E_stage1:.2e} kJ/mol")

        if E_stage1 < 1e6 and E_stage1 > 1e4:
            # Stage 2: Medium minimization
            print("\n  [Stage 2/3] Refining structure...")
            sim.minimizeEnergy(tolerance=100 * unit.kilojoule / (unit.mole * unit.nanometer), maxIterations=3000)

            state = sim.context.getState(getEnergy=True)
            E_stage2 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            print(f"    Energy: {E_stage2:.2e} kJ/mol")

        # Stage 3: Final minimization
        print("\n  [Stage 3/3] Final optimization...")
        sim.minimizeEnergy(tolerance=10 * unit.kilojoule / (unit.mole * unit.nanometer), maxIterations=5000)
    else:
        print("  Performing standard minimization...")
        sim.minimizeEnergy(tolerance=10 * unit.kilojoule / (unit.mole * unit.nanometer), maxIterations=10000)

    # Final energy
    state = sim.context.getState(getEnergy=True, getPositions=True)
    E_enzyme = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # Save minimized structure
    positions = state.getPositions()
    with open("enzyme_minimized.pdb", "w") as f:
        PDBFile.writeFile(pdb.topology, positions, f)
    print("\n[✓] Saved minimized enzyme → enzyme_minimized.pdb")

    # Results
    print(f"\n{'=' * 70}")
    print(f"ENERGY RESULTS")
    print(f"{'=' * 70}")
    print(f"Initial enzyme energy:  {E_initial:.2e} kJ/mol")
    print(f"Final enzyme energy:    {E_enzyme:.2f} kJ/mol")
    print(f"Ligand energy:          {E_ligand:.2f} kJ/mol")
    print(f"Energy reduction:       {E_initial - E_enzyme:.2e} kJ/mol")
    print(f"{'=' * 70}")

    # Interpret results
    if E_enzyme > 1e5:
        print("\n⚠ WARNING: Very high energy after minimization")
        print("  Your structure may still have problems")
    elif E_enzyme > 0:
        print("\n⚠ Positive energy (unfavorable)")
        print(f"  Reduced from {E_initial:.2e} kJ/mol")
        print("  Structure has some strain")
    else:
        print("\n✓ Negative energy (favorable)")

        if -50000 < E_enzyme < -5000:
            print(f"  ✓ Energy ({E_enzyme:.0f} kJ/mol) is in typical protein range")
        elif E_enzyme < -100000:
            print(f"  ⚠ Energy is very low ({E_enzyme:.0f} kJ/mol)")
            print("    May indicate force field artifacts")
        else:
            print(f"  Energy: {E_enzyme:.0f} kJ/mol")

    # Create complex for visualization
    print("\n[...] Creating enzyme-ligand complex for visualization...")
    pdb_minimized = PDBFile("enzyme_minimized.pdb")
    modeller = Modeller(pdb_minimized.topology, pdb_minimized.positions)

    pdb_lig = PDBFile(OUTPUT_LIGAND)
    modeller.add(pdb_lig.topology, pdb_lig.positions)

    with open("complex.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print("[✓] Saved complex → complex.pdb")

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & NEXT STEPS")
    print("=" * 70)
    print("\n✓ Enzyme structure processed successfully")
    print("✓ Files created:")
    print("  - enzyme_cleaned.pdb (with H atoms and caps)")
    print("  - enzyme_minimized.pdb (energy-minimized)")
    print("  - ligand.pdb (your urea molecule)")
    print("  - complex.pdb (enzyme + ligand for viewing)")
    print("\n⚠ Important: This does NOT calculate binding affinity!")
    print("\nFor binding energy calculations, use:")
    print("  1. AutoDock Vina - molecular docking (recommended)")
    print("  2. HADDOCK - protein-ligand docking")
    print("  3. MM-PBSA/GBSA - binding free energy")
    print("  4. FEP - most accurate, computationally expensive")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Error during energy calculation: {e}")
    import traceback

    traceback.print_exc()

    print("\n" + "=" * 70)
    print("TROUBLESHOOTING")
    print("=" * 70)
    print("If the error mentions missing atoms or templates:")
    print("  - Your AlphaFold structure may be incomplete")
    print("  - Try downloading the full structure from AlphaFold DB")
    print("  - Check if your structure has all chains/domains")
    print("=" * 70)