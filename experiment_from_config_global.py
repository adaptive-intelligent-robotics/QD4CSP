import json
import sys
from pathlib import Path

from ase.ga.utilities import CellBounds

from qd4csp.cli.map_elites_main import main
from qd4csp.crystal.materials_data_model import MaterialProperties, StartGenerators
from qd4csp.utils.experiment_parameters import ExperimentParameters


if __name__ == "__main__":
    file_location = ""
    if file_location == "":
        file_location = sys.argv[1]
    with open(file_location, "r") as file:
        experiment_parameters = json.load(file)

    experiment_parameters = ExperimentParameters(**experiment_parameters)
    experiment_parameters.cellbounds = (
        CellBounds(
            bounds={
                "phi": [20, 160],
                "chi": [20, 160],
                "psi": [20, 160],
                "a": [2, 40],
                "b": [2, 40],
                "c": [2, 40],
            }
        ),
    )
    experiment_parameters.splits = {(2,): 1, (4,): 1}
    experiment_parameters.cvt_run_parameters["behavioural_descriptors"] = [
        MaterialProperties(value)
        for value in experiment_parameters.cvt_run_parameters["behavioural_descriptors"]
    ]

    experiment_parameters.start_generator = StartGenerators(
        experiment_parameters.start_generator
    )
    main(experiment_parameters, hide_prints=False)
