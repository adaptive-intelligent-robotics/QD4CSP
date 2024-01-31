import warnings
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.build import niggli_reduce
from ase.ga.utilities import CellBounds
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from qd4csp.crystal.materials_data_model import BandGapEnum
from qd4csp.map_elites.elites_utils import Species
from qd4csp.property_calculators.parallel_relaxation.structure_optimizer import BatchedStructureOptimizer
from qd4csp.property_calculators.band_gap_calculator import BandGapCalculator
from qd4csp.property_calculators.shear_modulus_calculator import (
    ShearModulusCalculator,
)
warnings.simplefilter("ignore")


class CrystalEvaluator:
    def __init__(
        self,
        with_force_threshold=True,
        fmax_relaxation_convergence: float = 0.2,
        force_threshold_fmax: float = 1.0,
        cellbounds: Optional[CellBounds] = None,
        bd_normalisation: Union[List[Optional[Tuple[float, float]]]] = None,
    ):
        self.relaxer = BatchedStructureOptimizer(
            fmax_threshold=fmax_relaxation_convergence
        )
        if bd_normalisation is not None:
            band_gap_normalisation = (bd_normalisation[0][0], bd_normalisation[1][0])
            shear_modulus_normalisation = (
                bd_normalisation[0][1],
                bd_normalisation[1][1],
            )
        else:
            band_gap_normalisation, shear_modulus_normalisation = None, None

        self.band_gap_calculator = BandGapCalculator(band_gap_normalisation)

        self.shear_modulus_calculator = ShearModulusCalculator(
            shear_modulus_normalisation
        )
        self.with_force_threshold = with_force_threshold
        self.force_threshold_fmax = force_threshold_fmax
        self.cellbounds = (
            cellbounds
            if cellbounds is not None
            else CellBounds(
                bounds={
                    "phi": [20, 160],
                    "chi": [20, 160],
                    "psi": [20, 160],
                    "a": [2, 40],
                    "b": [2, 40],
                    "c": [2, 40],
                }
            )
        )

    def batch_compute_fitness_and_bd(
        self,
        list_of_atoms: List[Dict[str, np.ndarray]],
        n_relaxation_steps: int,
    ):
        list_of_atoms = [Atoms.fromdict(atoms) for atoms in list_of_atoms]
        kill_list = self._check_atoms_in_cellbounds(list_of_atoms)
        relaxation_results, updated_atoms = self.relaxer.relax(
            list_of_atoms, n_relaxation_steps
        )
        energies = -np.array(
            [
                relaxation_results[i]["trajectory"]["energies"]
                for i in range(len(relaxation_results))
            ]
        )
        structures = [
            relaxation_results[i]["final_structure"]
            for i in range(len(relaxation_results))
        ]
        if self.with_force_threshold:
            forces = np.array(
                [
                    relaxation_results[i]["trajectory"]["forces"]
                    for i in range(len(relaxation_results))
                ]
            )

            fitness_scores = self._apply_force_threshold(energies, forces)
        else:
            fitness_scores = energies
            forces = np.array(
                [
                    relaxation_results[i]["trajectory"]["forces"]
                    for i in range(len(relaxation_results))
                ]
            )

        band_gaps = self._batch_band_gap_compute(structures)
        shear_moduli = self._batch_shear_modulus_compute(structures)

        new_atoms_dict = [atoms.todict() for atoms in updated_atoms]

        for i in range(len(list_of_atoms)):
            new_atoms_dict[i]["info"] = list_of_atoms[i].info

        descriptors = (band_gaps, shear_moduli)

        del relaxation_results
        del structures
        del list_of_atoms

        return (
            updated_atoms,
            new_atoms_dict,
            fitness_scores,
            descriptors,
            kill_list,
            forces,
        )

    def batch_create_species(
        self, list_of_atoms, fitness_scores, descriptors, kill_list
    ):
        kill_list = np.array(kill_list)
        individual_indexes_to_add = np.arange(len(list_of_atoms))[~kill_list]
        species_list = []

        for i in individual_indexes_to_add:
            new_specie = Species(
                x=list_of_atoms[i],
                fitness=fitness_scores[i],
                desc=tuple([descriptors[j][i] for j in range(len(descriptors))]),
            )
            species_list.append(new_specie)

        return species_list

    def _batch_band_gap_compute(self, list_of_structures: List[Structure]):
        band_gaps = []
        for i in range(len(list_of_structures)):
            band_gap = self._compute_band_gap(
                relaxed_structure=list_of_structures[i]
            )
            band_gaps.append(band_gap)
        return band_gaps

    def _compute_band_gap(
        self,
        relaxed_structure: Structure,
        bandgap_type: BandGapEnum = BandGapEnum.SCAN,
    ):
        graph_attrs = torch.tensor([bandgap_type.value])
        bandgap = self.band_gap_calculator.compute(
            structure=relaxed_structure,
            band_gap_type=graph_attrs,
        )
        return float(bandgap)

    def _batch_shear_modulus_compute(self, list_of_structures: List[Structure]):
        shear_moduli = []
        for structure in list_of_structures:
            shear_modulus = self.shear_modulus_calculator.compute(
                structure,
            )
            shear_moduli.append(shear_modulus)
        return shear_moduli

    def _check_atoms_in_cellbounds(
        self,
        list_of_atoms: List[Atoms],
    ) -> List[bool]:
        kill_list = []
        for i, atoms in enumerate(list_of_atoms):
            if not self.cellbounds.is_within_bounds(atoms.get_cell()):
                niggli_reduce(atoms)
                if not self.cellbounds.is_within_bounds(atoms.get_cell()):
                    kill_individual = True
                else:
                    kill_individual = True
            else:
                kill_individual = False

            kill_list.append(kill_individual)
        return kill_list

    def _apply_force_threshold(
        self, energies: np.ndarray, forces: np.ndarray
    ) -> np.ndarray:
        fitnesses = np.array(energies)
        if self.with_force_threshold:
            fmax = self.compute_fmax(forces)
            indices_above_threshold = np.argwhere(
                fmax > self.force_threshold_fmax
            ).reshape(-1)
            forces_above_threshold = -1 * np.abs(
                fmax[fmax > self.force_threshold_fmax] - self.force_threshold_fmax
            )
            np.put(fitnesses, indices_above_threshold, forces_above_threshold)
        return fitnesses

    @staticmethod
    def compute_fmax(forces: np.ndarray):
        return np.max((forces**2).sum(axis=2), axis=1) ** 0.5

    def is_structure_valid_for_evaluation(self, structure: Union[Structure, Atoms]) -> bool:
        if isinstance(structure, Atoms):
            structure = AseAtomsAdaptor.get_structure(structure)

        return self.relaxer.is_structure_graph_valid(structure)
