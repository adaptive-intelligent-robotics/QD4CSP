import pickle
from argparse import ArgumentParser
from pathlib import Path

from qd4csp.evaluation.experiment_processing import ExperimentProcessor
from qd4csp.map_elites.cvt_centroids.initialise import __centroids_filename


def plot_all_metrics_and_cvt_for_experiment(
        experiment_path: Path,
        save_structure_images: bool = False,
        annotate: bool = False
):
    with open(experiment_path / "experiment_parameters.pkl", "rb") as file:
        experiment_parameters = pickle.load(file)

    bd_minimum_values, bd_maximum_values = experiment_parameters.return_min_max_bd_values()
    centroid_filename = __centroids_filename(
        k=experiment_parameters.number_of_niches,
        dim=experiment_parameters.n_behavioural_descriptor_dimensions,
        bd_names=experiment_parameters.cvt_run_parameters[
            "behavioural_descriptors"
        ],
        bd_minimum_values=bd_minimum_values,
        bd_maximum_values=bd_maximum_values,
        formula=experiment_parameters.system_name,
    )

    experiment_processor = ExperimentProcessor(
        experiment_label=experiment_path.name,
        config_filepath=experiment_parameters,
        centroid_filename=centroid_filename,
        fitness_limits=experiment_parameters.fitness_min_max_values,
        save_structure_images=save_structure_images,
        filter_for_experimental_structures=False,
        experiment_directory_path=experiment_path
    )

    # experiment_processor.plot(annotate=annotate)
    experiment_processor.process_symmetry(
        archive_number=experiment_parameters.maximum_evaluations,
        annotate=annotate,
    )


def plot_all_metrics_for_experiment_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        help="Path to experiment that requires plotting."
    )

    parser.add_argument(
        "-a",
        "--annotate",
        help="Optional. Annotate plots True / False.",
        default=False,
    )

    parser.add_argument(
        "-s",
        "--save_structures",
        help="Optional. Save structure images True / False",
        default=False,
    )

    args = parser.parse_args()
    plot_all_metrics_and_cvt_for_experiment(
        experiment_path=Path(args.experiment),
        annotate=args.annotate,
        save_structure_images=args.save_structures,
    )
