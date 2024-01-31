import os
import os
import tracemalloc

from ase.ga.utilities import CellBounds

from qd4csp.cli.map_elites_main import main
from qd4csp.crystal.materials_data_model import MaterialProperties, StartGenerators

from qd4csp.utils.experiment_parameters import ExperimentParameters


if __name__ == "__main__":
    import pyximport
    pyximport.install(setup_args={"script_args": ["--verbose"]})
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    experiment_parameters = ExperimentParameters(
        number_of_niches=200,
        maximum_evaluations=4,
        experiment_tag="debug_script",
        filter_comparison_data_for_n_atoms=24,
        start_generator=StartGenerators.PYXTAL,
        cvt_run_parameters={
            # more of this -> higher-quality CVT
            "cvt_samples": 25000,
            # we evaluate in batches to paralleliez
            "batch_size": 10,
            # proportion of niches to be filled before starting
            "random_init": 0,
            # batch for random initialization
            "random_init_batch": 1,
            # when to write results (one generation = one batch)
            "dump_period": 2,
            # do we use several cores?
            "parallel": True,
            # do we cache the result of CVT and reuse?
            "cvt_use_cache": True,
            # min/max of parameters
            "bd_minimum_values": (0, 0),
            "bd_maximum_values": (4, 120),
            "behavioural_descriptors": [
                MaterialProperties.BAND_GAP,
                MaterialProperties.SHEAR_MODULUS,
            ],
            "filter_starting_structures": 24,
            "seed": False,
            "profiling": False,
            "force_threshold": False,
            "force_threshold_exp_fmax": 2.0,
            "fmax_threshold": 0.2,
            "number_of_relaxation_steps":0,
            "normalise_bd": True,
        },
        system_name="TiO2",
        blocks=[22] * 8 + [8] * 16,
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
        operator_probabilities=[0.0, 0, 5.0, 5.0],
        ### CVT PARAMETERS ###
        n_behavioural_descriptor_dimensions=2,
        fitness_min_max_values=[0, 10],
    )
    main(experiment_parameters)
