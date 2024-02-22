import pickle
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

    experiment_processor.plot(annotate=annotate)
    experiment_processor.process_symmetry(
        archive_number=experiment_parameters.maximum_evaluations,
        annotate=annotate,
    )
