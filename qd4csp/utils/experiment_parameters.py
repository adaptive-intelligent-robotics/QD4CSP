import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional

from ase.ga.utilities import CellBounds

from qd4csp.crystal.materials_data_model import StartGenerators, MaterialProperties


@dataclass
class ExperimentParameters:
    system_name: str
    blocks: List[int]
    volume: int
    ratio_of_covalent_radii: float
    splits: Dict[Tuple[int], int]
    cellbounds: CellBounds
    operator_probabilities: List[float]
    n_behavioural_descriptor_dimensions: int
    number_of_niches: int
    maximum_evaluations: int
    cvt_run_parameters: Dict[str, Any]
    experiment_tag: str
    fitness_min_max_values: []
    filter_comparison_data_for_n_atoms: Optional[int]
    start_generator: StartGenerators
    """Contains all necessary parameters to run an experiment.
    
    Args:
        system_name: formula of your system e.g. TiO2.
        blocks: list of integers representing each atom in a cell 
            e.g. [22, 8, 8] represents a cell with 1 atom of Ti and 2 atoms of oxygen 
        volume: initial volume of unit cell. 
        ratio_of_covalent_radii: ratio to compute the minimum allowed distance 
            between atom centres
        splits: unit cell splits required for random structure generator
            these are unused if pyxtal generator is selected.
        cellbounds: cell boundary limits for optimisation. 
        operator_probabilities: probabilities of an operator being used in 
            order: cut_and_splice, soft_mutation, strain_mutation, 
            permutation_mutation.
        n_behavioural_descriptor_dimensions: number of dimensions in 
            behavioural descriptor (feature vector).
        number_of_niches: number of centroids in map.
        maximum_evaluations: maximum number of evaluations in MAP-Elites run. 
        cvt_run_parameters: dictionary containing all parameters for cvt experiment.
            explained in detail in default_cvt_run_parameters method.
        experiment_tag: tag to add to the experiment folder name
        fitness_min_max_values:: minimum values of fitness, used for plotting.
        filter_comparison_data_for_n_atoms: maximum number of atoms for 
            reference structures for plotting
        start_generator: type of start generator (random / pyxtal).
    """

    @classmethod
    def generate_default_to_populate(cls):
        return cls(
            system_name="populate_here",
            blocks="populate_here",
            volume=450,
            ratio_of_covalent_radii=0.4,
            splits={(2,): 1, (4,): 1},
            cellbounds=CellBounds(
                bounds={
                    "phi": [20, 160],
                    "chi": [20, 160],
                    "psi": [20, 160],
                    "a": [2, 40],
                    "b": [2, 40],
                    "c": [2, 40],
                }
            ),
            operator_probabilities=[0, 0, 5, 5],
            n_behavioural_descriptor_dimensions=2,
            number_of_niches=200,
            maximum_evaluations="populate_here",
            cvt_run_parameters=cls.default_cvt_run_parameters(),
            experiment_tag="populate_here",
            fitness_min_max_values="populate_here",
            filter_comparison_data_for_n_atoms=24,
            start_generator=StartGenerators.PYXTAL,
        )

    @staticmethod
    def default_cvt_run_parameters():
        return {
            # more of this -> higher-quality CVT
            "cvt_samples": 25000,
            # we evaluate in batches to paralleliez
            "batch_size": 1,
            # proportion of niches to be filled before starting
            "random_init": 0,
            # batch for random initialization
            "random_init_batch": 20,
            # when to write results (one generation = one batch)
            "dump_period": 2,
            # do we cache the result of CVT and reuse?
            "cvt_use_cache": True,
            # min/max of parameters
            "bd_minimum_values": (0, 0),
            "bd_maximum_values": (4, 120),
            "behavioural_descriptors": [
                MaterialProperties.BAND_GAP,
                MaterialProperties.SHEAR_MODULUS,
            ],  # currently only this combination is handled natively
            # number of relaxation for lal individuals
            "number_of_relaxation_steps": 0,
            # if seeding is used, the maximum number of atoms in structures to use
            "filter_starting_structures": 24,
            # whether to add reference structures into the search
            "seed": False,
            # whether to apply force threshold
            "force_threshold": False,
            # value of force threshold
            "force_threshold_exp_fmax": 2.0,
            # if whole archive is being relaxed
            "relax_archive_every_n_generations_n_relaxation_steps": 0,
            # fmax threshold for fire algorithm
            "fmax_threshold": 0.2,
        }

    def save_as_json(self, experiment_directory_path, filename: Optional[str]=None):
        self.splits = "DUMMY"
        self.cellbounds = "DUMMY"
        self.cvt_run_parameters["behavioural_descriptors"] = [
            descriptor.value
            for descriptor in self.cvt_run_parameters["behavioural_descriptors"]
        ]
        self.start_generator = self.start_generator.value
        self.blocks = list(self.blocks)

        filename = filename if filename is not None else "config"

        with open(f"{experiment_directory_path}/{filename}.json", "w") as file:
            json.dump(asdict(self), file)

    def return_min_max_bd_values(self):
        if self.cvt_run_parameters["normalise_bd"]:
            bd_minimum_values, bd_maximum_values = [0, 0], [1, 1]
        else:
            bd_minimum_values, bd_maximum_values = (
                self.cvt_run_parameters["bd_minimum_values"],
                self.cvt_run_parameters["bd_maximum_values"],
            )

        return bd_minimum_values, bd_maximum_values
