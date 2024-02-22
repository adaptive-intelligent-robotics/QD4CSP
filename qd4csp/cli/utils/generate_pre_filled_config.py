from argparse import ArgumentParser

from qd4csp.utils.env_variables import CONFIGS_FOLDER
from qd4csp.utils.experiment_parameters import ExperimentParameters


def generate_config():
    parser = ArgumentParser()

    parser.add_argument(
        "-f",
        "--filename",
        help="Name of your config file.",
        default="demo_config",
        type=str,
    )

    args = parser.parse_args()


    experiment_parameters = ExperimentParameters.generate_default_to_populate()
    experiment_parameters.save_as_json(
        experiment_directory_path=CONFIGS_FOLDER, filename=args.filename
    )
