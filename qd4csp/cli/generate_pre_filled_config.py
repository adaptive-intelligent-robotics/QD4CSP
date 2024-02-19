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

    parser.add_argument(
        "-s",
        "--subfolder",
        help="Name of subfolder in which config should be saved (within the main configs folder).",
        default="",
        type=str,
    )
    args = parser.parse_args()

    path_to_save_config = (
        CONFIGS_FOLDER / args.onfig_subfolder
    )
    path_to_save_config.mkdir(exist_ok=True)

    experiment_parameters = ExperimentParameters.generate_default_to_populate()
    experiment_parameters.save_as_json(
        experiment_directory_path=path_to_save_config, filename=args.filename
    )
