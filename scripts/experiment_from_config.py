import json
import sys
from pathlib import Path

from ase.ga.utilities import CellBounds

from qd4csp.cli.map_elites_main import main
from qd4csp.crystal.materials_data_model import MaterialProperties, \
    StartGenerators
from qd4csp.utils.experiment_parameters import ExperimentParameters

if __name__ == "__main__":
    file_location = ""
    if file_location == "":
        file_location = sys.argv[1]

    experiment_parameters = ExperimentParameters.from_config_json(file_location=Path(file_location))
    main(experiment_parameters, hide_prints=False)
