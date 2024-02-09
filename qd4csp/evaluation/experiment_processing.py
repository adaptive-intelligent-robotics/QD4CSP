import json
import os
import pathlib
from typing import Union, Optional

import numpy as np
from ase.ga.utilities import CellBounds
from tqdm import tqdm

from qd4csp.crystal.materials_data_model import MaterialProperties, \
    StartGenerators
from qd4csp.evaluation.plotting.figure_manager import FigureManager
from qd4csp.evaluation.plotting.plotter import CVTPlotting
from qd4csp.evaluation.plotting.plotting_data_model import CVTPlottingData, \
    PlotTypes
from qd4csp.evaluation.symmetry_evaluator import StructureEvaluation
from qd4csp.map_elites.archive import Archive
from qd4csp.utils import experiment_parameters
from qd4csp.utils.asign_target_values_to_centroids import (
    reassign_data_from_pkl_to_new_centroids,
)
from qd4csp.utils.env_variables import EXPERIMENT_FOLDER, MP_REFERENCE_FOLDER
from qd4csp.utils.experiment_parameters import ExperimentParameters
from qd4csp.utils.get_mpi_structures import get_all_materials_with_formula
from qd4csp.utils.utils import load_centroids, load_archive_from_pickle
from qd4csp.evaluation.plotting.utils import plot_all_statistics_from_file, \
    plot_gif

class ExperimentProcessor:
    def __init__(
        self,
        experiment_label: str,
        centroid_filename: str,
        config_filepath: Union[pathlib.Path, ExperimentParameters],
        fitness_limits=(6.5, 10),
        save_structure_images: bool = True,
        filter_for_experimental_structures: bool = False,
        centroid_directory_path: Optional[pathlib.Path] = None,
        experiment_directory_path: Optional[pathlib.Path] = None,
    ):
        self.experiment_label = experiment_label
        self.experiment_directory_path = (
            EXPERIMENT_FOLDER / experiment_label
            if experiment_directory_path is None
            else experiment_directory_path
        )
        self.centroid_directory_path = (
            EXPERIMENT_FOLDER / centroid_filename[1:]
            if centroid_directory_path is None
            else centroid_directory_path
        )
        self.all_centroids = load_centroids(str(self.centroid_directory_path))
        self.experiment_parameters = self._load_experiment_parameters(config_filepath)
        self.fitness_limits = fitness_limits
        self.save_structure_images = save_structure_images
        self.filter_for_experimental_structures = filter_for_experimental_structures
        self.formula = experiment_label[15:].split("_")[0]

        cvt_plot_limits = self.experiment_parameters.return_min_max_bd_values()

        self.plotter = CVTPlotting(
            centroids=self.all_centroids,
            target_centroids=self.compute_target_centroids(),
            axes_minimum_values=cvt_plot_limits[0],
            axes_maximum_values=cvt_plot_limits[1],
            directory_string=self.experiment_directory_path,
            overload_x_axis_limits=(
                self.experiment_parameters.cvt_run_parameters[
                    "bd_minimum_values"][0],
                self.experiment_parameters.cvt_run_parameters[
                    "bd_maximum_values"][0],
            ),
            overload_y_axis_limits=(
                self.experiment_parameters.cvt_run_parameters[
                    "bd_minimum_values"][1],
                self.experiment_parameters.cvt_run_parameters[
                    "bd_maximum_values"][1],
            ),
        )

    @staticmethod
    def _load_experiment_parameters(
        file_location: Union[pathlib.Path, ExperimentParameters]
    ):
        if isinstance(file_location, pathlib.Path):
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
                for value in experiment_parameters.cvt_run_parameters[
                    "behavioural_descriptors"
                ]
            ]

            experiment_parameters.start_generator = StartGenerators(
                experiment_parameters.start_generator
            )
        else:
            experiment_parameters = file_location

        return experiment_parameters

    def plot(self,
             annotate: bool = True, force_replot: bool = False):
        self._plot_all_maps_in_archive(annotate=annotate, force_replot=force_replot)
        plot_gif(experiment_directory_path=str(self.experiment_directory_path))
        plot_all_statistics_from_file(
            filename=f"{self.experiment_directory_path}/{self.experiment_label}.dat",
            save_location=f"{self.experiment_directory_path}/",
        )

    def _plot_all_maps_in_archive(self, annotate: bool = False, force_replot: bool = False):
        list_of_files = [
            name
            for name in os.listdir(f"{self.experiment_directory_path}")
            if not os.path.isdir(name)
        ]
        list_of_archives = [
            filename
            for filename in list_of_files
            if ("archive_" in filename) and (".pkl" in filename)
        ]
        list_of_plots = [
            filename
            for filename in list_of_files
            if ("cvt_plot" in filename) and (".png" in filename)
        ]
        list_of_plot_ids = [
            filename.lstrip("cvt_plot_").rstrip(".png") for filename in
            list_of_plots
        ]

        for filename in tqdm(list_of_archives, desc="Plotting cvt maps"):
            if "relaxed_archive" in filename:
                continue
            archive_id = filename.lstrip("relaxed_archive_").rstrip(".pkl")
            if force_replot or (archive_id not in list_of_plot_ids):
                cvt_plotting_data = CVTPlottingData.for_cvt_plot(
                    archive=Archive.from_archive(
                        archive_path=self.experiment_directory_path / filename,
                        centroid_filepath=self.centroid_directory_path,
                    ),
                    fitness_limits=self.experiment_parameters.fitness_min_max_values
                )
                self.plotter.plot(
                    ax=None,
                    fig=None,
                    plotting_metadata=cvt_plotting_data,
                    annotate=annotate,
                )

    def _get_last_archive_number(self):
        return max(
            [
                int(name.lstrip("archive_").rstrip(".pkl"))
                for name in os.listdir(self.experiment_directory_path)
                if (
                    (not os.path.isdir(name))
                    and ("archive_" in name)
                    and (".pkl" in name)
                )
            ]
        )

    def compute_target_centroids(self):
        number_of_atoms = len(self.experiment_parameters.blocks)
        bd_tag = [
            bd.value
            for bd in self.experiment_parameters.cvt_run_parameters[
                "behavioural_descriptors"
            ]
        ]
        tag = ""
        for el in bd_tag:
            tag += f"{el}_"
        comparison_data_location = (
            MP_REFERENCE_FOLDER
            / f"{self.formula}_{number_of_atoms}"
            / f"{self.formula}_{tag[:-1]}.pkl"
        )

        comparison_data_packed = load_archive_from_pickle(str(comparison_data_location))

        normalise_bd_values = (
            (
                self.experiment_parameters.cvt_run_parameters["bd_minimum_values"],
                self.experiment_parameters.cvt_run_parameters["bd_maximum_values"],
            )
            if self.experiment_parameters.cvt_run_parameters["normalise_bd"]
            else None
        )

        target_centroids = reassign_data_from_pkl_to_new_centroids(
            centroids_file=str(self.centroid_directory_path),
            target_data=comparison_data_packed,
            filter_for_number_of_atoms=self.experiment_parameters.filter_comparison_data_for_n_atoms,
            normalise_bd_values=normalise_bd_values,
        )
        return target_centroids

    def get_material_project_info(self):
        structure_info, known_structures = get_all_materials_with_formula(
            experiment_parameters.system_name
        )
        return structure_info, known_structures

    def process_symmetry(
        self,
        annotate=True,
        figure_manager: Optional[FigureManager] = None,
        archive_number: Optional[int] = None
    ):
        figure_manager = figure_manager if figure_manager is not None else FigureManager.create_empty()
        archive_number = archive_number if archive_number is not None else self._get_last_archive_number()
        unrelaxed_archive_location = (
            self.experiment_directory_path / f"archive_{archive_number}.pkl"
        )

        centroid_tag = str(self.centroid_directory_path.name).rstrip(".dat")
        number_of_atoms = self.experiment_parameters.cvt_run_parameters[
            "filter_starting_structures"
        ]
        target_data_path = (
            MP_REFERENCE_FOLDER
            / f"{self.formula}_{number_of_atoms}"
            / f"{self.formula}_target_data_{centroid_tag}.csv"
        )
        if not os.path.isfile(target_data_path):
            target_data_path = None

        archive = Archive.from_archive(
            unrelaxed_archive_location, centroid_filepath=self.centroid_directory_path
        )

        normalise_bd_values = (
            (
                self.experiment_parameters.cvt_run_parameters["bd_minimum_values"],
                self.experiment_parameters.cvt_run_parameters["bd_maximum_values"],
            )
            if self.experiment_parameters.cvt_run_parameters["normalise_bd"]
            else None
        )

        tareget_archive = Archive.from_reference_csv_path(
            target_data_path,
            normalise_bd_values=normalise_bd_values,
            centroids_path=self.centroid_directory_path,
        )
        symmetry_evaluation = StructureEvaluation(
            formula=self.formula,
            filter_for_experimental_structures=self.filter_for_experimental_structures,
            reference_data_archive=tareget_archive,
        )
        all_individual_indices_to_check = None

        df, individuals_with_matches = symmetry_evaluation.executive_summary_csv(
            archive=archive,
            indices_to_compare=all_individual_indices_to_check,
            directory_to_save=self.experiment_directory_path,
        )

        cvt_plot_data = CVTPlottingData.for_cvt_plot(
            archive=archive,
            fitness_limits=self.experiment_parameters.fitness_min_max_values
        )
        self.plotter.plot(
            ax=figure_manager.plot_to_axes_mapping[PlotTypes.CVT],
            annotate=annotate,
            fig=figure_manager.figure,
            plotting_metadata=cvt_plot_data,
        )

        list_of_centroid_groups, list_of_colours = \
            symmetry_evaluation.group_structures_by_symmetry(
                archive=archive,
                experiment_directory_path=self.experiment_directory_path,

            )

        group_data = CVTPlottingData.for_structure_groups(
            list_of_centroid_groups=list_of_centroid_groups,
            list_of_colours=list_of_colours,
        )
        self.plotter.plot(
            ax=figure_manager.plot_to_axes_mapping[PlotTypes.GROUPS],
            fig=figure_manager.figure,
            plotting_metadata=group_data,
            annotate=True,
        )

        if individuals_with_matches and target_data_path is not None:
            (
                plotting_from_archive,
                plotting_from_mp,
            ) = symmetry_evaluation.matches_for_plotting(individuals_with_matches)

            report_statistic_summary_dict = (
                symmetry_evaluation.write_report_summary_json(
                    plotting_matches_from_archive=plotting_from_archive,
                    plotting_matches_from_mp=plotting_from_mp,
                    directory_string=str(self.experiment_directory_path),
                )
            )

            energy_data = CVTPlottingData.for_energy_comparison(
                archive=archive,
                plotting_matches=plotting_from_archive,
                target_centroid_ids=np.array(
                    symmetry_evaluation.reference_data.loc["centroid_id"].array),
                target_centroid_energies=np.array(
                    symmetry_evaluation.reference_data.loc["energy"].array),
            )

            self.plotter.plot(
                ax=figure_manager.plot_to_axes_mapping[PlotTypes.ENERGY],
                annotate=annotate,
                fig=figure_manager.figure,
                plotting_metadata=energy_data,
            )

            unique_reference_match_data = CVTPlottingData.for_reference_matching(
                plotting_matches=plotting_from_mp,
                target_centroid_ids=symmetry_evaluation.reference_data.loc["centroid_id"].array,
                reference_shear_moduli=[symmetry_evaluation.reference_data.loc["shear_modulus"][ref] for ref in plotting_from_mp.mp_references],
                reference_band_gaps=[symmetry_evaluation.reference_data.loc["band_gap"][ref] for ref in plotting_from_mp.mp_references],
            )
            self.plotter.plot(
                ax=figure_manager.plot_to_axes_mapping[PlotTypes.UNIQUE_MATCHES],
                annotate=annotate,
                fig=figure_manager.figure,
                plotting_metadata=unique_reference_match_data,
            )

            all_reference_match_data = CVTPlottingData.for_reference_matching(
                plotting_matches=plotting_from_archive,
                target_centroid_ids=symmetry_evaluation.reference_data.loc["centroid_id"].array,
                reference_shear_moduli=None,
                reference_band_gaps=None,
            )

            self.plotter.plot(
                ax=figure_manager.plot_to_axes_mapping[PlotTypes.ALL_MATCHES],
                annotate=annotate,
                fig=figure_manager.figure,
                plotting_metadata=all_reference_match_data,
            )
