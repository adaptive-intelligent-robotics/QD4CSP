import copy
import json
import pathlib
import pickle
from collections import defaultdict
from typing import Optional, Tuple, List, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from ase import Atoms
from ase.spacegroup import get_spacegroup
from matplotlib import pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
from sklearn.neighbors import KDTree

from qd4csp.crystal.crystal_evaluator import CrystalEvaluator
from qd4csp.crystal.materials_data_model import MaterialProperties
from qd4csp.evaluation.plotting.plotter import CVTPlotting
from qd4csp.evaluation.plotting.plotting_data_model import CVTPlottingData
from qd4csp.evaluation.symmetry_evaluator import StructureEvaluation
from qd4csp.map_elites.archive import Archive
from qd4csp.map_elites.cvt_centroids.initialise import \
    __centroids_filename as get_centroids_filename, write_centroids, cvt, \
    initialise_kdt_and_centroids
from qd4csp.reference_setup.reference_plotter import ReferencePlotter
from qd4csp.utils.asign_target_values_to_centroids import (
    reassign_data_from_pkl_to_new_centroids,
)
from qd4csp.utils.env_variables import EXPERIMENT_FOLDER, MP_REFERENCE_FOLDER
from qd4csp.utils.experiment_parameters import ExperimentParameters
from qd4csp.utils.utils import load_centroids, normalise_between_0_and_1

import scienceplots

plt.style.use("science")
plt.rcParams["savefig.dpi"] = 300


class ReferenceAnalyser:
    def __init__(
        self,
        formula: str,
        max_n_atoms_in_cell: int,
        experimental_references_only: bool,
        save_plots: bool = True,
        normalise: bool = True,
        centroid_filename: Optional[str] = None,
    ):
        self.formula = formula
        self.max_n_atoms_in_cell = max_n_atoms_in_cell
        self.experimental_references_only = experimental_references_only
        self.experimental_string = (
            "experimental_only" if experimental_references_only else "exp_and_theory"
        )
        self.main_experiments_directory = EXPERIMENT_FOLDER
        self.symmetry_evaluator = StructureEvaluation(
            formula=formula,
            tolerance=0.1,
            maximum_number_of_atoms_in_reference=max_n_atoms_in_cell,
            number_of_atoms_in_system=max_n_atoms_in_cell,
            filter_for_experimental_structures=experimental_references_only,
        )
        self.structures_to_consider = sorted(
            self.symmetry_evaluator.known_structures_docs,
            key=lambda x: (x.theoretical, len(x.structure)),
        )
        self.crystal_evaluator = CrystalEvaluator(
            with_force_threshold=False,
            fmax_relaxation_convergence=0.2,
            force_threshold_fmax=1.0,
            bd_normalisation=None,
        )
        self.reference_ids = [
            str(structure.material_id) for structure in self.structures_to_consider
        ]
        self.behavioural_descriptors = [
            MaterialProperties.BAND_GAP,
            MaterialProperties.SHEAR_MODULUS,
        ]
        (
            self.energies,
            self.fmax_list,
            self.band_gaps,
            self.shear_moduli,
        ) = self.compute_target_values()
        self.centroid_filename = centroid_filename
        self.plotting_helper = ReferencePlotter(save_plots)
        self.save_plot = save_plots
        self.bd_minimum_values = None
        self.bd_maximum_values = None

        self.save_path = (
            MP_REFERENCE_FOLDER
            / f"{formula}_{max_n_atoms_in_cell}"
        )
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.normalise_bd = normalise

    def compute_target_values(self):
        list_of_atoms_as_dict = [
            AseAtomsAdaptor.get_atoms(el.structure).todict()
            for el in self.structures_to_consider
        ]
        atom_counts_per_structure = np.array(
            [len(el.structure.atomic_numbers) for el in self.structures_to_consider]
        )
        unique_lengths = np.unique(atom_counts_per_structure)
        energies, band_gaps, shear_moduli, forces, reference_ids_tracking = (
            [],
            [],
            [],
            [],
            [],
        )
        for el in unique_lengths:
            indices_to_check = np.argwhere(atom_counts_per_structure == el).reshape(-1)
            reference_ids_tracking += list(indices_to_check)
            structures_to_check = [list_of_atoms_as_dict[el] for el in indices_to_check]
            (
                _,
                _,
                fitness_scores,
                descriptors,
                _,
                structure_forces,
            ) = self.crystal_evaluator.batch_compute_fitness_and_bd(
                structures_to_check, n_relaxation_steps=0
            )
            energies += fitness_scores.tolist()
            band_gaps += descriptors[0]
            shear_moduli += descriptors[1]
            forces += structure_forces.tolist()

        indices_to_sort = np.argsort(reference_ids_tracking)

        fmax_list = []
        for force in forces:
            fmax_list.append(np.max((np.array(force[:-3]) ** 2).sum(axis=1) ** 0.5))

        fmax_list = np.take(fmax_list, indices_to_sort, axis=0)
        energies = np.take(energies, indices_to_sort, axis=0)
        shear_moduli = np.take(shear_moduli, indices_to_sort, axis=0)
        band_gaps = np.take(band_gaps, indices_to_sort, axis=0)

        return (
            np.array(energies),
            fmax_list,
            np.array(band_gaps),
            np.array(shear_moduli),
        )

    def set_bd_limits(self, band_gaps: np.ndarray, shear_moduli: np.ndarray):
        if band_gaps.min() == band_gaps.max():
            band_gap_limits = np.array([0.9 *band_gaps.min(), 1.1 * band_gaps.max()],
                                       dtype=float)
        else:
            band_gap_limits = np.array([band_gaps.min(), band_gaps.max()], dtype=float)

        if shear_moduli.min() == shear_moduli.max():
            shear_moduli_limits = np.array(
                [0.9 * shear_moduli.min(), 1.1 * shear_moduli.max()], dtype=float
            )
        else:
            shear_moduli_limits = np.array(
                [shear_moduli.min(), shear_moduli.max()], dtype=float
            )

        band_gap_min_max_diff = band_gap_limits[1] - band_gap_limits[0]
        shear_moduli_min_max_diff = shear_moduli_limits[1] - shear_moduli_limits[0]
        if not np.abs(shear_moduli_min_max_diff - band_gap_min_max_diff) < 0.2 * max(
            [shear_moduli_min_max_diff, band_gap_min_max_diff]
        ):
            print(
                f"Recommend setting band gaps manually, "
                f"bg limits: {band_gap_limits}, "
                f"shear moduli limits {shear_moduli_limits}"
            )

        self.bd_minimum_values = (band_gap_limits[0], shear_moduli_limits[0])
        self.bd_maximum_values = (band_gap_limits[1], shear_moduli_limits[1])
        return band_gap_limits.tolist(), shear_moduli_limits.tolist()

    def propose_fitness_limits(self):
        energies = np.array(self.energies)
        limits = np.array(
            [np.floor(energies.min() - 0.5), np.ceil(energies.max()) + 0.5]
        )
        return limits.tolist()

    def initialise_kdt_and_centroids(
        self,
        number_of_niches: int,
        band_gap_limits: Optional[np.ndarray] = None,
        shear_moduli_limits: Optional[np.ndarray] = None,
    ):
        # create the CVT
        if band_gap_limits is None or shear_moduli_limits is None:
            band_gap_limits, shear_moduli_limits = self.set_bd_limits(
                self.band_gaps, self.shear_moduli
            )
        if self.normalise_bd:
            bd_minimum_values, bd_maximum_values = [0, 0], [1, 1]
        else:
            bd_minimum_values = [band_gap_limits[0], shear_moduli_limits[0]]
            bd_maximum_values = [band_gap_limits[1], shear_moduli_limits[1]]
        kdt = initialise_kdt_and_centroids(
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
            experiment_directory_path=self.main_experiments_directory / "centroids",
            number_of_niches=number_of_niches,
            cvt_samples=25000,
            behavioural_descriptors_names=self.behavioural_descriptors,
            cvt_use_cache=True,
            formula=self.formula
        )

        self.centroid_filename = get_centroids_filename(
            k=number_of_niches,
            dim=len(self.behavioural_descriptors),
            bd_names=self.behavioural_descriptors,
            formula=self.formula,
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
        )
        return self.centroid_filename

    def create_model_archive(
        self,
        bd_minimum_values: np.ndarray,
        bd_maximum_values: np.ndarray,
        save_reference: bool = False
    ):
        if self.centroid_filename is None:
            raise ValueError("Set centroid_filename attribute or "
                             "run intitialise_and_write_centroids method")
        descriptors = np.array(
            [
                (self.band_gaps[i], self.shear_moduli[i])
                for i in range(len(self.band_gaps))
            ]
        )
        individuals = [
            AseAtomsAdaptor.get_atoms(el.structure)
            for el in self.structures_to_consider
        ]

        normalise_bd_values = (
            (bd_minimum_values, bd_maximum_values) if self.normalise_bd else None
        )

        centroids = reassign_data_from_pkl_to_new_centroids(
            centroids_file=self.main_experiments_directory
            / self.centroid_filename[1:],
            target_data=[self.energies, None, copy.deepcopy(descriptors), individuals],
            filter_for_number_of_atoms=None,
            normalise_bd_values=normalise_bd_values,
        )

        if normalise_bd_values is not None:
            band_gap = normalise_between_0_and_1(
                descriptors[:, 0], (normalise_bd_values[0][0], normalise_bd_values[1][0])
            )
            shear_modulus = normalise_between_0_and_1(
                descriptors[:, 1], (normalise_bd_values[0][1], normalise_bd_values[1][1])
            )
            descriptors = np.vstack([band_gap, shear_modulus]).T

        target_archive = Archive(
            fitnesses=np.array(self.energies),
            centroids=centroids,
            descriptors=np.array(descriptors),
            individuals=individuals,
            centroid_ids=None,
            labels=self.reference_ids,
        )
        target_archive.centroid_ids = Archive.assign_centroid_ids(
            centroids, self.main_experiments_directory / self.centroid_filename[1:]
        )

        if save_reference:
            all_data = []
            for i in range(len(self.energies)):
                one_data_point = []
                one_data_point.append(self.energies[i])
                one_data_point.append(centroids[i])
                one_data_point.append(
                    np.array([self.band_gaps[i], self.shear_moduli[i]])
                )
                one_data_point.append(individuals[i].todict())

                all_data.append(one_data_point)

            with open(
                self.save_path / f"{self.formula}_band_gap_shear_modulus.pkl", "wb"
            ) as file:
                pickle.dump(all_data, file)

            centroid_tag = str(self.centroid_filename[1:].split("/")[1].rstrip(".dat"))
            filename = f"{self.formula}_target_data_{centroid_tag}.csv"
            df = pd.DataFrame(
                [
                    self.reference_ids,
                    self.energies,
                    descriptors[:, 0],
                    descriptors[:, 1],
                    self.fmax_list,
                    target_archive.centroid_ids,
                ]
            )
            df.columns = df.iloc[0]
            df = df[1:]
            df = df.reset_index(drop=True)
            df.index = ["energy", "band_gap", "shear_modulus", "fmax", "centroid_id"] #todo: remove centroid_id
            df.to_csv(self.save_path / filename)

        return target_archive

    def plot_cvt_plot(
        self,
        target_archive: Archive,
        bd_minimum_values: np.ndarray,
        bd_maximum_values: np.ndarray,
        fitness_limits: np.ndarray,
        x_axis_limits=None,
        y_axis_limits=None,
    ):
        plotting_centroids = load_centroids(
            self.main_experiments_directory / self.centroid_filename[1:]
        )

        if self.save_plot:
            directory_string = self.save_path / "plots"
            directory_string.mkdir(exist_ok=True)
        else:
            directory_string = None

        self.plotter = CVTPlotting(
            centroids=plotting_centroids,
            axes_minimum_values=bd_minimum_values,
            axes_maximum_values=bd_maximum_values,
            directory_string=directory_string,
            overload_x_axis_limits=x_axis_limits,
            overload_y_axis_limits=y_axis_limits,
        )

        plotting_data = CVTPlottingData.for_cvt_plot(
            archive=target_archive,
            fitness_limits=tuple(fitness_limits)
        )
        plotting_data.filename = f"{self.formula}_cvt_plot_{self.experimental_string}_no_annotate"

        self.plotter.plot(
            ax=None, fig=None, plotting_metadata=plotting_data, annotate=False
        )

        list_of_centroid_groups, list_of_colours = self.symmetry_evaluator.group_structures_by_symmetry(
            archive=target_archive,
            experiment_directory_path=self.main_experiments_directory,
        )

        group_data = CVTPlottingData.for_structure_groups(
            list_of_centroid_groups=list_of_centroid_groups,
            list_of_colours=list_of_colours,
        )
        self.plotter.plot(
            ax=None,
            fig=None,
            plotting_metadata=group_data,
            annotate=True,
        )
        plt.clf()

    def plot_fmax(self, histogram_range: Optional[Tuple[int, int]] = None):
        params = {"figure.figsize": [3.5, 2.625]}
        mpl.rcParams.update(params)
        histogram_range = histogram_range if histogram_range is None else (0, 1)
        fig, ax = plt.subplots()
        ax.hist(self.fmax_list, range=histogram_range, bins=40)
        ax.set_xlabel("Maximum Force on an Atom")
        ax.set_ylabel("Structure Count")
        ax.set_title("Maximum Force on Atom for Reference Structures")
        if self.save_plot:
            fig.savefig(
                self.save_path / "plots"
                / f"{self.formula}_fmax_histogram_no_stress{self.experimental_string}.png",
                format="png",
            )
        else:
            fig.show()
        plt.clf()

    def plot_symmetries(self):
        params = {"figure.figsize": [3.5, 2.625]}
        mpl.rcParams["font.size"] = 8
        mpl.rcParams.update(params)
        fig, ax = plt.subplots()

        experimental = [el for el in self.structures_to_consider if not el.theoretical]
        labels = ["Theoretical", "Experimental"]

        all_group_information = []
        for structure_group in [self.structures_to_consider, experimental]:
            symmetries = defaultdict(list)
            for el in structure_group:
                symmetry = get_spacegroup(
                    AseAtomsAdaptor.get_atoms(el.structure), symprec=0.1
                ).symbol
                symmetries[symmetry].append(str(el.material_id))
            all_group_information.append(symmetries)

        for i, symmetries_in_group in enumerate(all_group_information):
            ax.bar(
                list(symmetries_in_group.keys()),
                [len(value) for value in symmetries_in_group.values()],
                label=labels[i],
            )

        ax.set_ylabel("Structure Count")
        ax.set_xlabel("Symmetry")
        ax.set_title("Symmetries Across Reference Structures")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.legend(prop={"size": 6})
        plt.tight_layout()
        if self.save_plot:
            fig.savefig(
                self.save_path / "plots"
                / f"{self.formula}_symmetries_histogram_{self.experimental_string}.png",
                format="png",
            )
        else:
            fig.show()
        plt.clf()

    def _plot_histogram_of_distances(self, distances):
        params = {"figure.figsize": [3.5, 2.625]}
        mpl.rcParams.update(params)
        distances = np.array(distances)
        minimum_distances = [np.min(el[el > 0.001]) for el in distances]

        fig, ax = plt.subplots()
        ax.hist(minimum_distances, range=(0, 0.5), bins=20)
        ax.set_xlabel("Minimum Non-Zero Cosine Distance Between Reference Structures")
        ax.set_ylabel("Number of Structures")
        ax.set_title(
            "Minimum Non-Zero Cosine Distance Between Reference Structures Across References"
        )

        if self.save_plot:
            fig.savefig(
                self.save_path / "plots"
                / f"{self.formula}_{self.max_n_atoms_in_cell}_distances_histogram_{self.experimental_string}.png",
                format="png",
            )
        else:
            fig.show()
        plt.clf()

    def heatmap_structure_matcher_distances(self, annotate: bool = True):
        if len(self.structures_to_consider) >= 15:
            params = {"figure.figsize": [5, 5], "font.size": 4}
        else:
            params = {"figure.figsize": [3.5, 3.5], "font.size": 8}
        mpl.rcParams.update(params)
        all_structure_matcher_matches = []
        distances = []

        for strucutre_1 in self.structures_to_consider:
            matches_for_structure_1 = []
            distances_for_structure_1 = []
            for structure_2 in self.structures_to_consider:
                matches_for_structure_1.append(
                    self.symmetry_evaluator.structure_matcher.fit(
                        strucutre_1.structure, structure_2.structure
                    )
                )
                if len(strucutre_1.structure) == len(structure_2.structure):
                    strucutre_1.structure.sort()
                    structure_2.structure.sort()
                    distance_to_known_structure = float(
                        self.symmetry_evaluator.comparator._compare_structure(
                            AseAtomsAdaptor.get_atoms(strucutre_1.structure),
                            AseAtomsAdaptor.get_atoms(structure_2.structure),
                        )
                    )
                else:
                    distance_to_known_structure = 1

                distances_for_structure_1.append(distance_to_known_structure)

            all_structure_matcher_matches.append(matches_for_structure_1)
            distances.append(distances_for_structure_1)

        im, cbar = self.plotting_helper.heatmap(
            np.array(distances),
            self.reference_ids,
            self.reference_ids,
            xlabel="Material Project Reference Numbers",
            ylabel="Material Project Reference Numbers",
            title="Cosine Distances Between Fingerprints of Reference Structures",
            cmap="YlGn",
            cbarlabel="Cosine Distance Between Structure Fingerprints",
        )
        if annotate:
            texts = self.plotting_helper.annotate_heatmap(im, valfmt="{x:.1f}")

        plt.tight_layout()
        if self.save_plot:
            plt.savefig(
                self.save_path / "plots"
                / f"{self.formula}_distances_heatmap_no_annotate_{self.experimental_string}.png",
                format="png",
            )
        else:
            plt.show()
        plt.clf()
        self.plotting_helper.heatmap(
            np.array(all_structure_matcher_matches),
            self.reference_ids,
            self.reference_ids,
            xlabel="Material Project Reference Numbers",
            ylabel="Material Project Reference Numbers",
            title="StructureMatcher Confusion Matrix",
            cmap="YlGn",
            cbarlabel="Match between structures True / False",
        )
        plt.tight_layout()
        if self.save_plot:
            plt.savefig(
                self.save_path / "plots"
                / f"{self.formula}_structure_matcher_heatmap_{self.experimental_string}.png",
                format="png",
            )
        else:
            plt.show()
        plt.clf()

        self._plot_histogram_of_distances(distances)

    def return_valid_spacegroups_for_pyxtal(
        self, elements: List[str], atoms_counts: List[int]
    ):
        my_crystal = pyxtal()

        all_possible_sg = []
        if self.experimental_references_only:
            structures_to_consider = (
                self.symmetry_evaluator.initialise_reference_structures(
                    self.formula, self.max_n_atoms_in_cell, False
                )
            )
        else:
            structures_to_consider = self.structures_to_consider
        for el in structures_to_consider:
            all_possible_sg.append(el.structure.get_space_group_info()[1])

        valid_spacegroups_for_combination = []
        for el in all_possible_sg:
            try:
                my_crystal.from_random(3, el, elements, atoms_counts)
            except Comp_CompatibilityError:
                continue
            valid_spacegroups_for_combination.append(el)

        number_of_atoms = sum(atoms_counts)
        with open(
            self.save_path
            / f"{self.formula}_{number_of_atoms}_allowed_symmetries.json",
            "w",
        ) as file:
            json.dump(valid_spacegroups_for_combination, file)
        return valid_spacegroups_for_combination

    def write_base_config(
        self,
        bd_minimum_values: np.ndarray,
        bd_maximum_values: np.ndarray,
        fitness_limits: np.ndarray,
    ):
        blocks = self._return_blocks_list()

        experiment_parameters = ExperimentParameters.generate_default_to_populate()
        experiment_parameters.system_name = self.formula
        experiment_parameters.blocks = list(blocks)
        try:
            experiment_parameters.cvt_run_parameters["bd_minimum_values"] = bd_minimum_values.tolist()
            experiment_parameters.cvt_run_parameters["bd_maximum_values"] = bd_maximum_values.tolist()
        except AttributeError:
            experiment_parameters.cvt_run_parameters["bd_minimum_values"] = list(bd_minimum_values)
            experiment_parameters.cvt_run_parameters["bd_maximum_values"] = list(bd_maximum_values)

        experiment_parameters.fitness_min_max_values = list(fitness_limits)
        experiment_parameters.save_as_json(self.save_path)

    def _return_blocks_list(self):
        temp_atoms = Atoms(self.formula)
        number_of_formula_units = int(
            self.max_n_atoms_in_cell / len(temp_atoms.get_atomic_numbers())
        )
        temp_atoms = Atoms(self.formula * number_of_formula_units)
        blocks = temp_atoms.get_atomic_numbers().tolist()
        blocks.sort()
        return blocks

    def return_descriptors(self):
        return np.array(
            [
                (self.band_gaps[i], self.shear_moduli[i])
                for i in range(len(self.band_gaps))
            ]
        )

    @classmethod
    def prepare_reference_data(
        cls,
        formula: str,
        elements_list: List[str],
        elements_counts_list: List[int],
        max_n_atoms_in_cell: int,
        experimental_references_only: bool = False,
        number_of_centroid_niches: int = 200,
        fitness_limits: Optional[Union[np.ndarray, List[float]]] = None,
        band_gap_limits: Optional[Union[np.ndarray, List[float]]] = None,
        shear_modulus_limits: Optional[Union[np.ndarray, List[float]]] = None,
    ):
        reference_analyser = cls(
            formula=formula,
            max_n_atoms_in_cell=max_n_atoms_in_cell,
            experimental_references_only=experimental_references_only,
            save_plots=True,
            normalise=True,
            centroid_filename=None,
        )
        reference_analyser.return_valid_spacegroups_for_pyxtal(
            elements=elements_list, atoms_counts=elements_counts_list,
        )

        if band_gap_limits is None and shear_modulus_limits is None:
            band_gap_limits, shear_modulus_limits = \
                reference_analyser.set_bd_limits(
                    reference_analyser.band_gaps, reference_analyser.shear_moduli
                )
        else:
            reference_analyser.bd_minimum_values = np.array(
                [band_gap_limits[0], shear_moduli_limits[0]])
            reference_analyser.bd_maximum_values = np.array(
                [band_gap_limits[1], shear_moduli_limits[1]])
        reference_analyser.initialise_kdt_and_centroids(
            number_of_niches=number_of_centroid_niches,
            band_gap_limits=band_gap_limits,
            shear_moduli_limits=shear_modulus_limits,
        )
        if fitness_limits is None:
            fitness_limits = reference_analyser.propose_fitness_limits()

        reference_analyser.write_base_config(
            bd_minimum_values=reference_analyser.bd_minimum_values,
            bd_maximum_values=reference_analyser.bd_maximum_values,
            fitness_limits=fitness_limits,
        )
        target_archive = reference_analyser.create_model_archive(
            bd_minimum_values=reference_analyser.bd_minimum_values,
            bd_maximum_values=reference_analyser.bd_maximum_values,
            save_reference=True,
        )
        reference_analyser.plot_cvt_plot(
            target_archive=target_archive,
            bd_minimum_values=np.array([0, 0]) if reference_analyser.normalise_bd else reference_analyser.bd_minimum_values,
            bd_maximum_values=np.array([1, 1]) if reference_analyser.normalise_bd else reference_analyser.bd_maximum_values,
            fitness_limits=fitness_limits,
            x_axis_limits=reference_analyser.bd_minimum_values,
            y_axis_limits=reference_analyser.bd_maximum_values,
        )
        reference_analyser.heatmap_structure_matcher_distances(annotate=False)
        reference_analyser.plot_symmetries()
        reference_analyser.plot_fmax()

        return reference_analyser





if __name__ == "__main__":
    # elements_list = [["C"], ["Si", "O"], ["Si", "C"]]
    # atoms_counts_list = [[24], [8, 16], [12, 12]]
    # formulas = ["C", "SiO2", "SiC",]

    fitness_limits = None
    band_gap_limits = None
    shear_moduli_limits = None

    elements_list = [["Si", "C"]]
    atoms_counts_list = [[12, 12]]
    formulas = ["SiC"]
    fitness_limits = [8.7, 9.5]
    band_gap_limits = [0, 4]
    shear_moduli_limits = [0, 120]

    for filter_experiment in [False]:
        for i, formula in enumerate(formulas):
            ReferenceAnalyser.prepare_reference_data(
                formula=formula,
                elements_list=elements_list[i],
                elements_counts_list=atoms_counts_list[i],
                max_n_atoms_in_cell=sum(atoms_counts_list[i]),
                experimental_references_only=filter_experiment,
                number_of_centroid_niches=200,
                fitness_limits=fitness_limits,
                band_gap_limits=band_gap_limits,
                shear_modulus_limits=shear_moduli_limits,
            )
