import os
import pickle
from pathlib import Path

from qd4csp.cli.all_plots_for_single_experiment import \
    plot_all_metrics_and_cvt_for_experiment
from qd4csp.utils.experiment_parameters import ExperimentParameters

if __name__ == '__main__':

    # experiment_parameters = ExperimentParameters.from_config_json(Path(__file__).parent.parent / "experiment_configs/TiO2_benchmark.json")
    #
    # base_path = Path(__file__).parent.parent / ".experiments.nosync/experiments/reported_results_update/TiO2/TiO2_benchmark_1"
    # directories = os.listdir(base_path)
    #
    # for directory in directories:
    #     if (base_path / directory).is_dir() and directory != ".DS_Store":
    #         experiment_parameters.dump_to_pickle(base_path / directory)
    #         with open(base_path / directory / "experiment_parameters.pkl", "rb") as file:
    #             experiment_parameters = pickle.load(file)

    # experiment_parameters = ExperimentParameters.from_config_json(Path(__file__).parent.parent / "experiment_configs/TiO2_benchmark.json")
    # experiment = Path(__file__).parent.parent / f".experiments.nosync/experiments/" \
    #                                             f"reported_results_update_backup/TiO2/TiO2_benchmark_with_high_threshold/20240201_09_04_TiO2_benchmark_1_9"
    # with open(experiment / "experiment_parameters.pkl", "wb") as file:
    #     experiment_parameters = pickle.dump(experiment_parameters, file)
    experiment = Path(__file__).parent.parent / f"experiments/20240221_14_20_TiO2_demo"
    plot_all_metrics_and_cvt_for_experiment(
        experiment
    )

# from qd4csp.cli.all_plots_for_single_experiment import plot_all_metrics_and_cvt_for_experiment
# from pathlib import Path
# experiment = Path("experiments/20240221_14_20_TiO2_demo")
# plot_all_metrics_and_cvt_for_experiment(experiment)
