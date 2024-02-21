import json
from typing import List, Dict, Tuple, Optional

import numpy as np
from ase import Atoms
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.offspring_creator import OperationSelector
from ase.ga.soft_mutation import SoftMutation
from ase.ga.standardmutations import StrainMutation, PermutationMutation, \
    RattleMutation
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import CellBounds, closest_distances_generator
from chgnet.graph import CrystalGraphConverter
from pymatgen.io.ase import AseAtomsAdaptor
from pyxtal import pyxtal

from qd4csp.crystal.materials_data_model import StartGenerators
from qd4csp.map_elites.elites_utils import Species
from qd4csp.utils.env_variables import EXPERIMENT_FOLDER, MP_REFERENCE_FOLDER


class CrystalSystem:
    def __init__(
        self,
        atom_numbers_to_optimise: List[float],
        volume: int = 450,
        ratio_of_covalent_radii: float = 0.4,
        splits: Dict[Tuple[int], int] = None,
        cellbounds: CellBounds = None,
        operator_probabilities: List[float] = (0.0, 0.0, 5.0, 5.0),
        compound_formula: str = "TiO2",
        start_generator: StartGenerators = StartGenerators.RANDOM,
    ):
        self.volume = volume
        self.atom_numbers_to_optimise = atom_numbers_to_optimise
        self.atomic_numbers = np.unique(np.array(self.atom_numbers_to_optimise))
        self.ratio_of_covalent_radii = ratio_of_covalent_radii
        self.splits = splits if splits is not None else {(2,): 1, (1,): 1}
        self.cellbounds = (
            cellbounds
            if cellbounds is not None
            else CellBounds(
                bounds={
                    "phi": [20, 160],
                    "chi": [20, 160],
                    "psi": [20, 160],
                    "a": [2, 60],
                    "b": [2, 60],
                    "c": [2, 60],
                }
            )
        )
        self._start_generator = self._initialise_start_generator(start_generator)
        self._strain_mutation = None
        self._cut_and_splice = None
        self._soft_mutation = None
        self._permutation_mutation = None

        self.operators = self._initialise_operators(operator_probabilities)

        self.compound_formula = compound_formula
        self.main_experiment_directory = EXPERIMENT_FOLDER
        self._possible_pyxtal_modes = self.load_possible_pyxtal_spacegroups()

        self.graph_converter = CrystalGraphConverter(on_isolated_atoms="warn")

    def load_possible_pyxtal_spacegroups(self):
        reference_tag = (
            f"{self.compound_formula}_{len(self.atom_numbers_to_optimise)}"
        )
        with open(
            MP_REFERENCE_FOLDER
            / reference_tag
            / f"{reference_tag}_allowed_symmetries.json",
            "r",
        ) as file:
            valid_spacegroups_for_combination = json.load(file)

        return valid_spacegroups_for_combination

    def create_one_individual(self, individual_id: Optional[int]):
        if isinstance(self._start_generator, StartGenerator):
            try:
                individual = self._start_generator.get_new_candidate()
            except AssertionError:
                individual = self._start_generator.get_new_candidate()

        elif isinstance(self._start_generator, pyxtal):
            generate_structure = True
            while generate_structure:
                species, counts = np.unique(
                    Atoms(self.atom_numbers_to_optimise).get_chemical_symbols(),
                    return_counts=True,
                )
                self._start_generator.from_random(
                    dim=3,
                    group=np.random.choice(self._possible_pyxtal_modes),
                    species=species.tolist(),
                    numIons=counts.tolist(),
                )
                generate_structure = not self._start_generator.valid
            individual = AseAtomsAdaptor.get_atoms(self._start_generator.to_pymatgen())

        individual.info["confid"] = individual_id
        return individual

    def create_n_individuals(
        self, number_of_individuals: int
    ) -> List[Dict[str, np.ndarray]]:
        individuals = []
        for i in range(number_of_individuals):
            new_individual = self.create_one_individual(individual_id=i)
            try:
                if (
                    self.graph_converter(
                        AseAtomsAdaptor.get_structure(atoms=new_individual),
                        # on_isolated_atoms="warn",
                    )
                    is not None
                ):
                    new_individual = new_individual.todict()
                    individuals.append(new_individual)
            except SystemExit:
                continue
        return individuals

    def _initialise_start_generator(self, start_generator: StartGenerators):
        if start_generator == StartGenerators.RANDOM:
            closest_distances = closest_distances_generator(
                atom_numbers=self.atom_numbers_to_optimise,
                ratio_of_covalent_radii=self.ratio_of_covalent_radii,
            )  # equivalent to blmin
            return StartGenerator(
                Atoms("", pbc=True),
                self.atom_numbers_to_optimise,
                closest_distances,
                box_volume=self.volume,
                number_of_variable_cell_vectors=3,
                cellbounds=self.cellbounds,
                splits=self.splits,
            )
        elif start_generator == StartGenerators.PYXTAL:
            return pyxtal()
        else:
            raise NotImplemented("Pick a valid start generator (random or pyxtal).")

    def _initialise_operators(self, operator_probabilities: List[float]):
        closest_distances = closest_distances_generator(
            atom_numbers=self.atomic_numbers,
            ratio_of_covalent_radii=self.ratio_of_covalent_radii,
        )

        self._cut_and_splice = CutAndSplicePairing(
            Atoms("", pbc=True),
            len(self.atom_numbers_to_optimise),
            closest_distances,
            p1=1.0,
            p2=0.0,
            minfrac=0.15,
            number_of_variable_cell_vectors=3,
            cellbounds=self.cellbounds,
            use_tags=False,
        )

        self._strain_mutation = StrainMutation(
            closest_distances,
            stddev=0.7,
            cellbounds=self.cellbounds,
            number_of_variable_cell_vectors=3,
            use_tags=False,
        )

        closest_distances_soft_mutation = closest_distances_generator(
            self.atom_numbers_to_optimise, 0.1
        )
        self._soft_mutation = SoftMutation(
            closest_distances_soft_mutation,
            bounds=[2.0, 5.0],
            use_tags=False,
        )

        self._permutation_mutation = PermutationMutation(
            len(self.atom_numbers_to_optimise)
        )
        return OperationSelector(
            operator_probabilities,
            [
                self._cut_and_splice,
                self._soft_mutation,
                self._strain_mutation,
                self._permutation_mutation,
            ],
        )

    def _initialise_alternative_operators(
        self, alternative_operators,
    ):
        closest_distances = closest_distances_generator(
            atom_numbers=self.atomic_numbers,
            ratio_of_covalent_radii=self.ratio_of_covalent_radii,
        )
        operator_probabilities = []
        operator_list = []
        for operator, probability in alternative_operators:
            if operator == "rattle":
                self._rattle_mutation = RattleMutation(
                    blmin=closest_distances,
                    n_top=len(self.atomic_numbers),
                )
                operator_list.append(self._rattle_mutation)

            operator_probabilities.append(probability)

        return OperationSelector(
            operator_probabilities,
            operator_list,
        )

    def mutate(self, parents: List[Species]) -> Atoms:
        mutator = self.operators.get_operator()
        new_individual, _ = mutator.get_new_individual(
            [Atoms.fromdict(parents[0].x), Atoms.fromdict(parents[1].x)]
        )
        return new_individual

    def update_operator_scaling_volumes(self, population: List[Atoms]):
        if self._strain_mutation in self.operators.oplist:
            self._strain_mutation.update_scaling_volume(
                population, w_adapt=0.5, n_adapt=4
            )
        if self._cut_and_splice in self.operators.oplist:
            self._cut_and_splice.update_scaling_volume(
                population, w_adapt=0.5, n_adapt=4
            )
