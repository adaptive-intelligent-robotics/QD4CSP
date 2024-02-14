import logging
from pathlib import Path

from qd4csp.reference_setup.reference_analyser import ReferenceAnalyser
from reproduce_results.all_plots_for_single_experiment import \
    plot_all_metrics_and_cvt_for_experiment
from reproduce_results.average_statistics_generator import \
    AverageStatisticsGenerator

if __name__ == '__main__':
    print("Plotting mean TiO2 results: Figure 3")
    average_statistics_generator = AverageStatisticsGenerator(
        path_to_experiments=Path(__file__).parent
        / f".experiments.nosync/reported_results/TiO2/",
    )
    average_statistics_generator.plot_mean_statistics(
        folder_names=["TiO2_benchmark_1"],
        labels=None,
        filter_coverage_for_valid_solutions_only=False,
        plot_individually=True,
    )

    print("Computing average match performance for TiO2: section 4.2")
    average_statistics_generator.compute_average_match_performance_for_all_experiments()

    print("Plotting mean TiO2 results: Figures 4, 5, 7")
    plot_all_metrics_and_cvt_for_experiment(
        Path(__file__).parent / f".experiments.nosync/reported_results/TiO2/TiO2_benchmark_1/20240201_09_04_TiO2_benchmark_1_9"
    )

    print("Computing mean statistics for other materials: Table 1")
    average_statistics_generator = AverageStatisticsGenerator(
        path_to_experiments=Path(__file__).parent
        / f".experiments.nosync/reported_results/SiO2/",
    )
    average_statistics_generator.compute_average_match_performance_for_all_experiments()

    average_statistics_generator = AverageStatisticsGenerator(
        path_to_experiments=Path(__file__).parent
        / f".experiments.nosync/reported_results/SiC/",
    )
    average_statistics_generator.compute_average_match_performance_for_all_experiments()

    average_statistics_generator = AverageStatisticsGenerator(
        path_to_experiments=Path(__file__).parent
        / f".experiments.nosync/reported_results/C/",
    )
    average_statistics_generator.compute_average_match_performance_for_all_experiments()

    print("Plotting Reference TiO2 information: Figure 2 + ESI")

    ReferenceAnalyser.prepare_reference_data(
        formula="TiO2",
        elements_list=["Ti", "O"],
        elements_counts_list=[8, 16],
        max_n_atoms_in_cell=24,
        experimental_references_only=False,
        number_of_centroid_niches=200,
        fitness_limits=[8.7, 9.5],
        band_gap_limits=[0, 4],
        shear_modulus_limits=[0, 120],
    )
    print("Plotting CVT plots for ESI materials (might take a while)")
    plot_all_metrics_and_cvt_for_experiment(
        Path(__file__).parent
        / f".experiments.nosync/reported_results/SiO2/"
          f"SiO2_like_benchmark_with_high_threshold/20240202_02_31_SiO2_SiO2_like_benchmark_with_high_threshold_1",
    )
    plot_all_metrics_and_cvt_for_experiment(
        Path(__file__).parent /
        f".experiments.nosync/reported_results/C/"
        f"C_like_benchmark_with_high_threshold/20240204_22_37_C_C_like_benchmark_with_high_threshold_10",
    )
    plot_all_metrics_and_cvt_for_experiment(
        Path(__file__).parent /
        f".experiments.nosync/reported_results/SiC/"
        f"SiC_like_benchmark_with_high_threshold/20240202_01_53_SiC_SiC_like_benchmark_with_high_threshold_10",
    )

    (Path(__file__).parent / f".experiments.nosync/reported_results/reported_results_generated").mkdir(exist_ok=True)
