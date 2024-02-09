from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
import torch
from ase import Atoms
from chgnet.model import CHGNet
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from qd4csp.property_calculators.parallel_relaxation.fire import OverridenFire
from qd4csp.property_calculators.parallel_relaxation.unit_cell_filter import AtomsFilterForRelaxation


class BatchedStructureOptimizer:
    def __init__(self, batch_size: int = 10, fmax_threshold: float = 0.2):
        self.overriden_optimizer = OverridenFire()
        self.atoms_filter = AtomsFilterForRelaxation()
        self.model = CHGNet.load()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.fmax_threshold = fmax_threshold

    def relax(
        self, list_of_atoms: List[Atoms], n_relaxation_steps: int, verbose: bool = False
    ) -> Tuple[List[Dict[str, np.ndarray]], List[Atoms]]:
        all_relaxed = False
        v = None
        Nsteps = 0
        dt = np.full(len(list_of_atoms), 0.1)
        a = np.full(len(list_of_atoms), 0.1)
        n_relax_steps = np.zeros(len(list_of_atoms))
        fmax_over_time = []

        converged_atoms = list(np.zeros(len(list_of_atoms)))
        atom_list_indices = list(np.arange(len(list_of_atoms)))
        original_cells = np.array([atoms.cell.array for atoms in list_of_atoms])

        while not all_relaxed:
            forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)

            forces, _ = self.atoms_filter.get_forces_exp_cell_filter(
                forces_from_chgnet=forces,
                stresses_from_chgnet=stresses,
                list_of_atoms=list_of_atoms,
                original_cells=original_cells,
                current_atom_cells=[atoms.cell.array for atoms in list_of_atoms],
                cell_factors=np.array([1] * len(list_of_atoms)),
            )
            if n_relaxation_steps == 0:
                all_relaxed = True
            else:
                fmax = np.max((forces**2).sum(axis=2), axis=1) ** 0.5
                fmax_over_time.append(fmax)
                if verbose:
                    print(Nsteps, energies * 24, fmax)

                v, dt, n_relax_steps, a, dr = self.overriden_optimizer.step_override(
                    forces, v, dt, n_relax_steps, a
                )

                positions = self.atoms_filter.get_positions(
                    original_cells,
                    [atoms.cell for atoms in list_of_atoms],
                    list_of_atoms,
                    np.array([1] * len(list_of_atoms)),
                )

                list_of_atoms = self.atoms_filter.set_positions(
                    original_cells,
                    list_of_atoms,
                    np.array(positions + dr),
                    np.array([1] * len(list_of_atoms)),
                )

                converged_mask = self.overriden_optimizer.converged(
                    forces, self.fmax_threshold
                )
                converged_atoms_indices = np.argwhere(converged_mask).reshape(-1)

                if bool(list(converged_atoms_indices)):
                    true_indices = []
                    for atom_index in converged_atoms_indices:
                        true_atom_index = atom_list_indices[atom_index]
                        true_indices.append(true_atom_index)
                        converged_atoms[true_atom_index] = list_of_atoms[atom_index]

                    list_of_atoms = [
                        el
                        for i, el in enumerate(list_of_atoms)
                        if i not in converged_atoms_indices
                    ]
                    atom_list_indices = [
                        el
                        for i, el in enumerate(atom_list_indices)
                        if i not in converged_atoms_indices
                    ]
                    v = np.array(
                        [
                            el
                            for i, el in enumerate(v)
                            if i not in converged_atoms_indices
                        ]
                    )

                    dt = np.array(
                        [
                            el
                            for i, el in enumerate(dt)
                            if i not in converged_atoms_indices
                        ]
                    )
                    a = np.array(
                        [
                            el
                            for i, el in enumerate(a)
                            if i not in converged_atoms_indices
                        ]
                    )
                    n_relax_steps = np.array(
                        [
                            el
                            for i, el in enumerate(n_relax_steps)
                            if i not in converged_atoms_indices
                        ]
                    )
                    original_cells = np.array(
                        [
                            el
                            for i, el in enumerate(original_cells)
                            if i not in converged_atoms_indices
                        ]
                    )

                Nsteps += 1
                all_relaxed = self._end_relaxation(
                    Nsteps, n_relaxation_steps, converged_mask
                )
                if all_relaxed:
                    if list_of_atoms:
                        remaining_indices = atom_list_indices
                        for atom_index, atom_object in zip(
                            remaining_indices, list_of_atoms
                        ):
                            # true_atom_index = atom_list_indices[atom_index]
                            converged_atoms[atom_index] = atom_object
                    list_of_atoms = converged_atoms

        final_structures = [
            AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms
        ]
        if n_relaxation_steps != 0:
            forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
        reformated_output = []
        for i in range(len(final_structures)):
            reformated_output.append(
                {
                    "final_structure": final_structures[i],
                    "trajectory": {
                        "energies": energies[i],
                        "forces": forces[i],
                        "stresses": stresses[i],
                    },
                }
            )

        return reformated_output, list_of_atoms

    def _evaluate_list_of_atoms(
        self, list_of_atoms: List[Atoms],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(list_of_atoms[0], Atoms):
            list_of_structures = [
                AseAtomsAdaptor.get_structure(list_of_atoms[i])
                for i in range(len(list_of_atoms))
            ]
        elif isinstance(list_of_atoms[0], Structure):
            list_of_structures = list_of_atoms

        hotfix_graphs = False
        indices_to_update = []
        try:
            graphs = [
                self.model.graph_converter(struct, on_isolated_atoms="warn")
                for struct in list_of_structures
            ]
        except SystemExit:
            graphs = []
            for i, struct in enumerate(list_of_structures):
                try:
                    graphs.append(
                        self.model.graph_converter(struct, on_isolated_atoms="warn")
                    )
                except SystemExit:
                    hotfix_graphs = True
                    indices_to_update.append(i)

        if None in graphs:
            hotfix_graphs = True
            new_graphs = []
            for i in range(len(graphs)):
                if graphs[i] is None:
                    indices_to_update.append(i)
                else:
                    new_graphs.append(graphs[i])
            graphs = new_graphs

            print(f"graphs end length {len(graphs)}")

        predictions = self.model.predict_graph(
            graphs,
            task="efs",
            return_atom_feas=False,
            return_crystal_feas=False,
            batch_size=self.batch_size,
        )
        if isinstance(predictions, dict):
            predictions = [predictions]

        forces = np.array([pred["f"] for pred in predictions])
        energies = np.array([pred["e"] for pred in predictions])
        stresses = np.array([pred["s"] for pred in predictions])

        if hotfix_graphs:
            for i in indices_to_update:
                forces = np.insert(forces, i, np.full((24, 3), 100), axis=0)
                energies = np.insert(energies, i, 10000)
                stresses = np.insert(stresses, i, np.full((3, 3), 100), axis=0)

        return forces, energies, stresses

    @staticmethod
    def _end_relaxation(nsteps: int, max_steps: int, forces_mask: np.ndarray):
        return (nsteps > max_steps) or forces_mask.all()

    def is_structure_graph_valid(self, structure: Structure) -> bool:
        try:
            self.model.graph_converter(structure, on_isolated_atoms="warn")
            return True
        except SystemExit:
            return False
