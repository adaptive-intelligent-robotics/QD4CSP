import shutil

from qd4csp.reference_setup.reference_analyser import ReferenceAnalyser
from qd4csp.utils.env_variables import MP_REFERENCE_FOLDER, EXPERIMENT_FOLDER
from qd4csp.cli.utils.all_plots_for_single_experiment import \
    plot_all_metrics_and_cvt_for_experiment
from qd4csp.evaluation.average_statistics_generator import \
    AverageStatisticsGenerator

def reproduce_results():
    path_to_reproduce_results = EXPERIMENT_FOLDER / "reported_results"
    (path_to_reproduce_results / "reported_figures").mkdir(exist_ok=True)

    print("Plotting mean TiO2 results: Figure 3")
    average_statistics_generator = AverageStatisticsGenerator(
        path_to_experiments=path_to_reproduce_results / "TiO2",
    )
    average_statistics_generator.plot_mean_statistics(
        folder_names=["TiO2_benchmark_with_high_threshold"],
        labels=None,
        filter_coverage_for_valid_solutions_only=False,
        plot_individually=True,
    )

    shutil.copy(
        src=path_to_reproduce_results / "TiO2/all_plots/tio2_benchmark_with_high_threshold_qd_score.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_3a.png",
    )
    shutil.copy(
        src=path_to_reproduce_results / "TiO2/all_plots/tio2_benchmark_with_high_threshold_coverage.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_3b.png",
    )
    shutil.copy(
        src=path_to_reproduce_results / "TiO2/all_plots/tio2_benchmark_with_high_threshold_median_fitness.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_3c.png",
    )
    shutil.copy(
        src=path_to_reproduce_results / "TiO2/all_plots/tio2_benchmark_with_high_threshold_maximum_fitness.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_3d.png",
    )


    print("Plotting mean TiO2 results: Figures 4, 5, 7")
    plot_all_metrics_and_cvt_for_experiment(
        path_to_reproduce_results / "TiO2/TiO2_benchmark_with_high_threshold/20240201_09_04_TiO2_benchmark_1_9"
    )

    shutil.copy(
        src=path_to_reproduce_results / "TiO2/TiO2_benchmark_with_high_threshold/20240201_09_04_TiO2_benchmark_1_9/cvt_plot_5000.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_4.png",
    )

    shutil.copy(
        src=path_to_reproduce_results / "TiO2/TiO2_benchmark_with_high_threshold/20240201_09_04_TiO2_benchmark_1_9/cvt_matches_from_archive_archive_matches_view.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_5a.png",
    )

    shutil.copy(
        src=path_to_reproduce_results / "TiO2/TiO2_benchmark_with_high_threshold/20240201_09_04_TiO2_benchmark_1_9/cvt_energy_diff_matches_from_archive_archive_matches_view.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_5b.png",
    )

    shutil.copy(
        src=path_to_reproduce_results / "TiO2/TiO2_benchmark_with_high_threshold/20240201_09_04_TiO2_benchmark_1_9/cvt_matches_from_archive_mp_reference_view.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_5c.png",
    )

    shutil.copy(
        src=path_to_reproduce_results / "TiO2/TiO2_benchmark_with_high_threshold/20240201_09_04_TiO2_benchmark_1_9/cvt_matches_from_archive_mp_reference_view.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_7.png",
    )

    print("Computing average match performance for TiO2: section 4.2")
    average_statistics_generator.compute_average_match_performance_for_all_experiments()

    shutil.copy(
        src=path_to_reproduce_results / "TiO2/TiO2_benchmark_with_high_threshold/experiment_performance.json",
        dst=path_to_reproduce_results / "reported_figures/main_section_4_2_tio2_performance.json",
    )

    print("Computing mean statistics for other materials: Table 1")
    for material in ["SiO2", "SiC", "C"]:
        average_statistics_generator = AverageStatisticsGenerator(
            path_to_experiments=path_to_reproduce_results / material,
        )
        average_statistics_generator.compute_average_match_performance_for_all_experiments()

        shutil.copy(
            src=path_to_reproduce_results / f"{material}/{material}_like_benchmark_with_high_threshold/experiment_performance.json",
            dst=path_to_reproduce_results / f"reported_figures/main_table_1_{material}_performance.json",
        )

    print("Plotting CVT plots for ESI materials (might take a while)")
    random_experiment_mapping = {
        "SiO2": "20240202_02_31_SiO2_SiO2_like_benchmark_with_high_threshold_1",
        "SiC": "20240202_01_53_SiC_SiC_like_benchmark_with_high_threshold_10",
        "C": "20240204_22_37_C_C_like_benchmark_with_high_threshold_10",
    }
    for material, experiment in random_experiment_mapping.items():
        plot_all_metrics_and_cvt_for_experiment(
            path_to_reproduce_results / f"{material}/{material}_like_benchmark_with_high_threshold/{experiment}",
        )

        shutil.copy(
            src=path_to_reproduce_results /
                f"{material}/{material}_like_benchmark_with_high_threshold/{experiment}/"
                f"cvt_plot_5000.png",
            dst=path_to_reproduce_results / f"reported_figures/esi_figure_S3_{material}_a_map_elites.png",
        )

        shutil.copy(
            src=path_to_reproduce_results /
                f"{material}/{material}_like_benchmark_with_high_threshold/{experiment}/"
                f"cvt_energy_diff_matches_from_archive_archive_matches_view.png",
            dst=path_to_reproduce_results / f"reported_figures/esi_figure_S3_{material}_b_energy_diff.png",
        )

        shutil.copy(
            src=path_to_reproduce_results /
                f"{material}/{material}_like_benchmark_with_high_threshold/{experiment}/"
                f"cvt_matches_from_archive_archive_matches_view.png",
            dst=path_to_reproduce_results / f"reported_figures/esi_figure_S3_{material}_c_archive_view.png",
        )

        shutil.copy(
            src=path_to_reproduce_results /
                f"{material}/{material}_like_benchmark_with_high_threshold/{experiment}/"
                f"cvt_matches_from_archive_mp_reference_view.png",
            dst=path_to_reproduce_results / f"reported_figures/esi_figure_S3_{material}_d_reference_view.png",
        )

        shutil.copy(
            src=path_to_reproduce_results /
                f"{material}/{material}_like_benchmark_with_high_threshold/{experiment}/"
                f"cvt_by_structure_similarity.png",
            dst=path_to_reproduce_results / f"reported_figures/esi_figure_S3_{material}_e_similarity.png",
        )

    print("Plotting Reference TiO2 information: Figure 2 + ESI")

    for experimental_only in [True, False]:
        ReferenceAnalyser.prepare_reference_data(
            formula="TiO2",
            elements_list=["Ti", "O"],
            elements_counts_list=[8, 16],
            max_n_atoms_in_cell=24,
            experimental_references_only=experimental_only,
            number_of_centroid_niches=200,
            fitness_limits=[8.7, 9.5],
            band_gap_limits=[0, 4],
            shear_modulus_limits=[0, 120],
        )

    shutil.copy(
        src=MP_REFERENCE_FOLDER / "TiO2_24/plots/TiO2_cvt_plot_exp_and_theory_no_annotate.png",
        dst=path_to_reproduce_results / "reported_figures/main_figure_2.png",
    )

    shutil.copy(
        src=MP_REFERENCE_FOLDER / "TiO2_24/plots/TiO2_fmax_histogram_exp_and_theory.png",
        dst=path_to_reproduce_results / "reported_figures/esi_figure_s1.png",
    )

    shutil.copy(
        src=MP_REFERENCE_FOLDER / "TiO2_24/plots/TiO2_structure_matcher_heatmap_experimental_only.png",
        dst=path_to_reproduce_results / "reported_figures/esi_figure_s2a.png",
    )

    shutil.copy(
        src=MP_REFERENCE_FOLDER / "TiO2_24/plots/TiO2_structure_matcher_heatmap_exp_and_theory.png",
        dst=path_to_reproduce_results / "reported_figures/esi_figure_s2b.png",
    )

    shutil.copy(
        src=MP_REFERENCE_FOLDER / "TiO2_24/plots/TiO2_structure_matcher_heatmap_experimental_only.png",
        dst=path_to_reproduce_results / "reported_figures/esi_figure_s2a.png",
    )

    with open(path_to_reproduce_results / "reported_figures/readme.txt", "w") as file:
        file.write(
            "For convenience this folder contains copies of figures used in this work. "
            "These can be also inspected directly in the following folders. \n"
            "\n"
            "TiO2 averaged: reproduce_results/TiO2/all_plots/ \n"
            "TiO2 archive: reproduce_results/TiO2/TiO2_benchmark_with_high_threshold/ \n"
            "TiO2 structures (cif files + images): reproduce_results/TiO2/TiO2_benchmark_with_high_threshold/ \n"
            "TiO2 reference info: mp_reference_analysis/TiO2_24/plots \n"
            "SiO2 archive: reproduce_results/SiO2/SiO2_like_benchmark_with_high_threshold/20240202_02_31_SiO2_SiO2_like_benchmark_with_high_threshold_1 \n"
            "SiC archive: reproduce_results/SiC/SiC_like_benchmark_with_high_threshold/20240202_01_53_SiC_SiC_like_benchmark_with_high_threshold_10 \n"
            "C archive: reproduce_results/C/C_like_benchmark_with_high_threshold/20240204_22_37_C_C_like_benchmark_with_high_threshold_10 \n"

        )

    print("All generated plots are copied to the reported_results/reported_figures folder. Please see reported_figures/readme.txt to inspect individual folders.")
