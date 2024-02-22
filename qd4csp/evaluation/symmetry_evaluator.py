import json
import os
import pathlib
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from ase import Atoms
from ase.ga.ofp_comparator import OFPComparator
from ase.spacegroup import get_spacegroup
from matplotlib import cm
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.vis.structure_vtk import StructureVis
from tqdm import tqdm

from qd4csp.evaluation.confidence_levels import ConfidenceLevels
from qd4csp.evaluation.plotting.plotting_data_model import PlottingMode, \
    PlottingMatches
from qd4csp.map_elites.archive import Archive
from qd4csp.utils.get_mpi_structures import get_all_materials_with_formula


class StructureEvaluation:
    def __init__(
        self,
        formula: str = "TiO2",
        tolerance: float = 0.1,
        maximum_number_of_atoms_in_reference: int = 24,
        number_of_atoms_in_system: int = 24,
        filter_for_experimental_structures: bool = True,
        reference_data_archive: Optional[Archive] = None,
        list_of_fooled_ground_state: Optional[List[str]] = None,
        enable_structure_visualiser: bool = False
    ):
        self.known_structures_docs = self.initialise_reference_structures(
            formula,
            maximum_number_of_atoms_in_reference,
            filter_for_experimental_structures,
        )
        self.material_ids = [
            self.known_structures_docs[i].material_id
            for i in range(len(self.known_structures_docs))
        ]

        self.known_space_group_spglib = self._get_reference_spacegroups(tolerance)
        self.tolerance = tolerance
        self.comparator = OFPComparator(
            n_top=number_of_atoms_in_system,
            dE=1.0,
            cos_dist_max=1e-3,
            rcut=10.0,
            binwidth=0.05,
            pbc=[True, True, True],
            sigma=0.05,
            nsigma=4,
            recalculate=False,
        )
        self.enable_structure_visualiser = enable_structure_visualiser
        if enable_structure_visualiser:
            self.structure_viewer = StructureVis(show_polyhedron=False, show_bonds=True)
        else:
            self.structure_viewer = None
        self.structure_matcher = StructureMatcher()
        self.fingerprint_distance_threshold = 0.1
        self.reference_data = (
            reference_data_archive.to_dataframe()
            if reference_data_archive is not None
            else None
        )
        try:
            self.ground_state = np.array(self.material_ids)[np.array([el.is_stable for el in self.known_structures_docs])][0]
        except IndexError:
            self.ground_state = None

        self.fooled_ground_states_dict = {
            "TiO2": ["mp-34688"],
            "SiO2": ['mp-10851', 'mp-7000', 'mp-6930', 'mp-6922'],
            "SiC": [],
            "C": [],
        }

        if formula not in list(self.fooled_ground_states_dict.keys()):
            self.fooled_ground_states_dict.update({formula: list_of_fooled_ground_state})

        self.fooled_ground_states = self.fooled_ground_states_dict[formula]

    def initialise_reference_structures(
        self,
        formula: str = "TiO2",
        max_number_of_atoms: int = 12,
        experimental: bool = True,
    ):
        docs, atom_objects = get_all_materials_with_formula(formula)
        if experimental:
            experimentally_observed = [
                docs[i]
                for i in range(len(docs))
                if (not docs[i].theoretical)
                and (len(docs[i].structure) <= max_number_of_atoms)
            ]

            return experimentally_observed
        else:
            references_filtered_for_size = [
                docs[i]
                for i in range(len(docs))
                if (len(docs[i].structure) <= max_number_of_atoms)
            ]

            return references_filtered_for_size

    def find_individuals_with_reference_symmetries(
        self,
        individuals: List[Atoms],
    ):
        spacegroup_dictionary = self.compute_symmetries_from_individuals(
            individuals, None
        )

        archive_to_reference_mapping = defaultdict(list)
        for key in self.known_space_group_spglib:
            if key in spacegroup_dictionary.keys():
                archive_to_reference_mapping[key] += spacegroup_dictionary[key]

        return archive_to_reference_mapping, spacegroup_dictionary

    def compute_symmetries_from_individuals(
        self,
        individuals: List[Atoms],
        archive_indices_to_check: Optional[List[int]] = None,
    ) -> Dict[str, List[int]]:
        if archive_indices_to_check is None:
            archive_indices_to_check = range(len(individuals))

        spacegroup_dictionary = defaultdict(list)
        for index in archive_indices_to_check:
            spacegroup = self.get_spacegroup_for_individual(individuals[index])
            if spacegroup is not None:
                spacegroup_dictionary[spacegroup].append(index)

        return spacegroup_dictionary

    def get_spacegroup_for_individual(self, individual: Atoms):
        for symprec in [self.tolerance]:
            try:
                spacegroup = get_spacegroup(individual, symprec=symprec).symbol
            except RuntimeError:
                spacegroup = None
        return spacegroup

    def _get_reference_spacegroups(self, tolerance: float):
        spglib_spacegroups = []
        for el in self.known_structures_docs:
            spglib_spacegroups.append(
                get_spacegroup(
                    AseAtomsAdaptor.get_atoms(el.structure), symprec=tolerance
                ).symbol
            )
        return spglib_spacegroups

    def _get_indices_to_check(self, fitnesses: np.ndarray) -> np.ndarray:
        return np.argwhere(fitnesses > self.tolerance).reshape(-1)

    def plot_histogram(
        self,
        spacegroup_dictionary: Dict[str, List[int]],
        against_reference: bool = True,
        save_directory: Optional[pathlib.Path] = None,
    ):
        if against_reference:
            spacegroup_counts = []
            for key in self.known_space_group_spglib:
                if key in spacegroup_dictionary.keys():
                    spacegroup_counts.append(len(spacegroup_dictionary[key]))
                else:
                    spacegroup_counts.append(0)
        else:
            spacegroups = list(spacegroup_dictionary.keys())
            spacegroup_counts = [len(value) for value in spacegroup_dictionary.values()]

        plt.bar(spacegroups, spacegroup_counts)
        plt.ylabel("Count of individuals in structure")
        plt.xlabel("Symmetry type computed using spglib")

        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_directory is None:
            plt.show()
        else:
            reference_string = "with_ref" if against_reference else ""
            plt.savefig(
                save_directory / f"ind_symmetries_histogram_{reference_string}.png",
                format="png",
            )

    def save_structure_visualisations(
        self,
        archive: Archive,
        structure_indices: List[int],
        directory_to_save: pathlib.Path,
        file_tags: List[str],
        save_primitive: bool = False,
    ) -> str:
        primitive_string = "_primitive" if save_primitive else ""
        filename = None
        for i, individual_index in enumerate(structure_indices):
            individual_as_structure = AseAtomsAdaptor.get_structure(
                archive.individuals[individual_index]
            )
            if save_primitive:
                individual_as_structure = SpacegroupAnalyzer(
                    structure=individual_as_structure
                ).find_primitive()

            individual_confid = archive.individuals[individual_index].info["confid"]

            filename = f"ind_{archive.centroid_ids[individual_index]}_{file_tags[i]}_{individual_confid}{primitive_string}"
            if self.enable_structure_visualiser:
                self.structure_viewer.set_structure(individual_as_structure)
                self.structure_viewer.write_image(
                    str(directory_to_save / f"{filename}.png" ),
                    magnification=5,
                )
            individual_as_structure.to(
                filename=str(directory_to_save / f"cif_{filename}.cif"))
        return filename

    def save_best_structures_by_energy(
        self,
        archive: Archive,
        fitness_range: Optional[Tuple[float, float]],
        top_n_individuals_to_save: int,
        directory_to_save: pathlib.Path,
        save_primitive: bool = False,
        save_visuals: bool = True,
    ) -> List[int]:
        sorting_indices = np.argsort(archive.fitnesses)
        sorting_indices = np.flipud(sorting_indices)

        top_n_individuals = sorting_indices[:top_n_individuals_to_save]
        if fitness_range is not None:
            individuals_in_fitness_range = np.argwhere(
                (archive.fitnesses >= fitness_range[0])
                * (archive.fitnesses <= fitness_range[1])
            ).reshape(-1)
        else:
            individuals_in_fitness_range = np.array([], dtype="int64")

        indices_to_check = np.unique(
            np.hstack([top_n_individuals, individuals_in_fitness_range]),
        )
        if save_visuals:
            self.save_structure_visualisations(
                archive=archive,
                structure_indices=list(indices_to_check),
                directory_to_save=directory_to_save,
                file_tags=["best_by_energy"],
                save_primitive=save_primitive,
            )
        return list(indices_to_check)

    def save_best_structures_by_symmetry(
        self,
        archive: Archive,
        matched_space_group_dict: Optional[Dict[str, np.ndarray]],
        directory_to_save: pathlib.Path,
        save_primitive: bool = False,
        save_visuals: bool = True,
    ) -> List[int]:
        if matched_space_group_dict is None:
            (
                space_group_matching_dict,
                _,
            ) = self.find_individuals_with_reference_symmetries(
                archive.individuals,
                None,
            )

        if save_visuals:
            for desired_symmetry, indices_to_check in matched_space_group_dict.items():
                self.save_structure_visualisations(
                    archive=archive,
                    structure_indices=list(indices_to_check),
                    directory_to_save=directory_to_save,
                    file_tag=f"best_by_symmetry_{desired_symmetry}",
                    save_primitive=save_primitive,
                )

        structure_indices = []
        for el in list(matched_space_group_dict.values()):
            structure_indices += el

        return structure_indices

    def make_symmetry_to_material_id_dict(self):
        spacegroup_to_reference_dictionary = defaultdict(list)
        for el in self.known_structures_docs:
            spacegroup = self.get_spacegroup_for_individual(
                AseAtomsAdaptor.get_atoms(el.structure)
            )
            spacegroup_to_reference_dictionary[spacegroup].append(str(el.material_id))
        return spacegroup_to_reference_dictionary

    def executive_summary_csv(
        self,
        archive: Archive,
        indices_to_compare: Optional[List[int]],
        directory_to_save: pathlib.Path,
        reference_data_path: Optional[pathlib.Path] = None,
    ) -> pd.DataFrame:
        summary_data = []

        symmetry_to_material_id_dict = self.make_symmetry_to_material_id_dict()
        if indices_to_compare is None:
            indices_to_compare = list(range(len(archive.individuals)))

        for structure_index in tqdm(indices_to_compare, desc="Evaluating individuals' structures"):
            structure = AseAtomsAdaptor.get_structure(
                archive.individuals[structure_index]
            )

            try:
                primitive_structure = SpacegroupAnalyzer(structure).find_primitive()
            except TypeError:
                continue
            spacegroup = self.get_spacegroup_for_individual(
                archive.individuals[structure_index]
            )
            symmetry_match = (
                symmetry_to_material_id_dict[spacegroup]
                if spacegroup in symmetry_to_material_id_dict.keys()
                else []
            )

            summary_row = {
                "individual_confid": archive.individuals[structure_index].info[
                    "confid"
                ],
                "centroid_index": archive.centroid_ids[structure_index],
                "fitness": archive.fitnesses[structure_index],
                "descriptors": archive.descriptors[structure_index],
                "symmetry": spacegroup,
                "number_of_cells_in_primitive_cell": len(primitive_structure),
                "matches": [],
            }

            for known_structure in self.known_structures_docs:
                reference_id = str(known_structure.material_id)
                structure_matcher_match = self.structure_matcher.fit(
                    structure, known_structure.structure
                )

                distance_to_known_structure = self._compute_distance(
                    primitive_structure,
                    known_structure.structure,
                )

                if (
                    distance_to_known_structure <= 0.1
                    or structure_matcher_match
                    or str(reference_id) in symmetry_match
                ):
                    if self.reference_data is not None:
                        ref_band_gap = self.reference_data[reference_id]["band_gap"]
                        ref_shear_modulus = self.reference_data[reference_id][
                            "shear_modulus"
                        ]
                        ref_energy = self.reference_data[reference_id]["energy"]
                        ref_centroid = self.reference_data[reference_id]["centroid_id"]
                        error_to_bg = (
                            (ref_band_gap - archive.descriptors[structure_index][0])
                            / ref_band_gap
                            * 100
                        )
                        error_to_shear = (
                            (
                                ref_shear_modulus
                                - archive.descriptors[structure_index][1]
                            )
                            / ref_shear_modulus
                            * 100
                        )
                        error_to_energy = (
                            (ref_energy - archive.fitnesses[structure_index])
                            / ref_energy
                            * 100
                        )
                        distance_in_bd_space = np.sqrt(
                            (ref_band_gap - archive.descriptors[structure_index][0])
                            ** 2
                            + (
                                ref_shear_modulus
                                - archive.descriptors[structure_index][1]
                            )
                            ** 2
                        )
                    else:
                        (
                            error_to_energy,
                            error_to_bg,
                            error_to_shear,
                            ref_centroid,
                            distance_in_bd_space,
                        ) = (None, None, None, None, None)
                    summary_row["matches"].append(
                        {
                            reference_id: {
                                "symmetry": reference_id in symmetry_match,
                                "structure_matcher": structure_matcher_match,
                                "distance": distance_to_known_structure,
                                "centroid": ref_centroid,
                                "reference_energy_perc_difference": error_to_energy,
                                "reference_band_gap_perc_difference": error_to_bg,
                                "reference_shear_modulus_perc_difference": error_to_shear,
                                "euclidian_distance_in_bd_space": distance_in_bd_space,
                            }
                        }
                    )

            summary_data.append(summary_row)
        individuals_with_matches = [
            individual for individual in summary_data if individual["matches"]
        ]

        df = pd.DataFrame(individuals_with_matches)
        try:
            df = df.explode("matches")
            df["matches"] = df["matches"].apply(lambda x: x.items())
            df = df.explode("matches")
        except KeyError:
            print("No matches found")
        df[["reference", "match_info"]] = pd.DataFrame(
            df["matches"].tolist(), index=df.index
        )
        df.drop(columns="matches", inplace=True)
        df = pd.concat([df, df["match_info"].apply(pd.Series)], axis=1)
        df.drop(columns="match_info", inplace=True)
        df.to_csv(directory_to_save / "ind_executive_summary.csv")
        return df, individuals_with_matches

    def matches_for_plotting(
        self, individuals_with_matches
    ) -> Tuple[PlottingMatches, PlottingMatches]:
        centroids_with_matches = []
        mp_reference_of_matches = []
        confidence_levels = []
        euclidian_distance_to_matches = []
        all_descriptors = []
        true_centroid_indices = []
        energy_difference = []
        for i, individual in enumerate(individuals_with_matches):
            sorted_matches = sorted(
                individual["matches"],
                key=lambda x: list(x.values())[0]["euclidian_distance_in_bd_space"],
            )
            match = sorted_matches[0]
            match_dictionary = list(match.values())[0]
            centroids_with_matches.append(int(individual["centroid_index"]))
            mp_reference_of_matches.append(list(match.keys())[0])
            confidence_levels.append(
                self._assign_confidence_level_in_match(
                    match_dictionary, individual["centroid_index"]
                )
            )
            euclidian_distance_to_matches.append(
                match_dictionary["euclidian_distance_in_bd_space"]
            )
            all_descriptors.append(individual["descriptors"])
            true_centroid_indices.append(match_dictionary["centroid"])
            energy_difference.append(
                match_dictionary["reference_energy_perc_difference"]
            )

        plotting_matches_from_archive = PlottingMatches(
            centroids_with_matches,
            mp_reference_of_matches,
            confidence_levels,
            euclidian_distance_to_matches,
            all_descriptors,
            energy_difference,
            PlottingMode.ARCHIVE_MATCHES_VIEW,
        )

        unique_matches, counts = np.unique(mp_reference_of_matches, return_counts=True)

        ref_centroids_with_matches = []
        ref_mp_reference_of_matches = []
        ref_confidence_levels = []
        ref_euclidian_distance_to_matches = []
        ref_all_descriptors = []
        ref_energy_difference = []

        for match_mp_ref in unique_matches:
            match_indices = np.argwhere(
                np.array(mp_reference_of_matches) == match_mp_ref
            ).reshape(-1)
            confidence_levels_for_ref = [
                confidence_levels[i].value for i in match_indices
            ]
            best_confidence_indices = np.argwhere(
                confidence_levels_for_ref == np.max(confidence_levels_for_ref)
            ).reshape(-1)
            match_indices = np.take(match_indices, best_confidence_indices)
            euclidian_distances = [
                euclidian_distance_to_matches[i] for i in match_indices
            ]
            euclidian_distances = np.array(euclidian_distances)

            closest_euclidian_distance_index = np.argwhere(
                euclidian_distances == np.min(euclidian_distances)
            ).reshape(-1)
            best_match_index = match_indices[closest_euclidian_distance_index]
            # print(f"{match_mp_ref} {best_match_index}")

            index_in_archive_list = best_match_index[0]
            ref_centroids_with_matches.append(
                int(true_centroid_indices[index_in_archive_list])
            )
            ref_mp_reference_of_matches.append(match_mp_ref)
            ref_confidence_levels.append(confidence_levels[index_in_archive_list])
            ref_euclidian_distance_to_matches.append(
                euclidian_distance_to_matches[index_in_archive_list]
            )
            ref_all_descriptors.append(all_descriptors[index_in_archive_list])
            ref_energy_difference.append(energy_difference[index_in_archive_list])

        plotting_matches_from_mp = PlottingMatches(
            ref_centroids_with_matches,
            ref_mp_reference_of_matches,
            ref_confidence_levels,
            ref_euclidian_distance_to_matches,
            ref_all_descriptors,
            ref_energy_difference,
            PlottingMode.MP_REFERENCE_VIEW,
        )

        return plotting_matches_from_archive, plotting_matches_from_mp

    def write_report_summary_json(
        self,
        plotting_matches_from_archive: PlottingMatches,
        plotting_matches_from_mp: PlottingMatches,
        directory_string: Union[pathlib.Path, str],
    ):
        if bool(self.ground_state in plotting_matches_from_archive.mp_references):
            indices_to_check = np.argwhere(
                np.array(plotting_matches_from_archive.mp_references) == self.ground_state
            ).reshape(-1)
            confidence_scores = np.take(
                plotting_matches_from_archive.confidence_levels, indices_to_check
            ).reshape(-1)
            ground_state_match = max(confidence_scores)
        else:
            ground_state_match = ConfidenceLevels.NO_MATCH

        fooled_ground_state_match = ConfidenceLevels.NO_MATCH
        for el in self.fooled_ground_states:
            if bool(el in plotting_matches_from_archive.mp_references):
                indices_to_check = np.argwhere(
                    np.array(plotting_matches_from_archive.mp_references) == el
                ).reshape(-1)
                confidence_scores = np.take(
                    plotting_matches_from_archive.confidence_levels, indices_to_check
                ).reshape(-1)
                fooled_ground_state_match = max(max(confidence_scores), fooled_ground_state_match.value)

        summary_dict = {
            "ground_state_match": ConfidenceLevels.get_string(
                ConfidenceLevels(ground_state_match)
            ),
            "fooled_ground_state_match": ConfidenceLevels.get_string(
                ConfidenceLevels(fooled_ground_state_match)
            ),
            "unique_reference_matches": len(
                np.unique(plotting_matches_from_archive.mp_references)
            ),

            "number_gold_unique":
                len(
                    np.argwhere(
                        np.array(plotting_matches_from_mp.confidence_levels)
                        == ConfidenceLevels.GOLD.value
                    ),
            ),

            "number_high_unique":
                len(
                    np.argwhere(
                        np.array(plotting_matches_from_mp.confidence_levels)
                        == ConfidenceLevels.HIGH.value
                    ),
                ),

            "number_medium_unique":
                len(
                    np.argwhere(
                        np.array(plotting_matches_from_mp.confidence_levels)
                        == ConfidenceLevels.MEDIUM.value
                    ),
                ),
            "number_low_unique":
                len(
                    np.argwhere(
                        np.array(plotting_matches_from_mp.confidence_levels)
                        == ConfidenceLevels.LOW.value
                    ),
                ),
            "number_gold": len(
                np.argwhere(
                    np.array(plotting_matches_from_archive.confidence_levels)
                    == ConfidenceLevels.GOLD.value
                )
            ),
            "number_high": len(
                np.argwhere(
                    np.array(plotting_matches_from_archive.confidence_levels)
                    == ConfidenceLevels.HIGH.value
                )
            ),
            "number_medium": len(
                np.argwhere(
                    np.array(plotting_matches_from_archive.confidence_levels)
                    == ConfidenceLevels.MEDIUM.value
                )
            ),
            "number_low": len(
                np.argwhere(
                    np.array(plotting_matches_from_archive.confidence_levels)
                    == ConfidenceLevels.LOW.value
                )
            ),
            "total_matches": len(plotting_matches_from_archive.mp_references),
        }
        with open(f"{directory_string}/ind_report_summary.json", "w") as file:
            json.dump(summary_dict, file)
        return summary_dict


    def _get_maximum_confidence_for_centroid_id(
        self, centroid_indices_to_check: np.ndarray, plotting_matches: PlottingMatches
    ):
        for centroid_id in centroid_indices_to_check:
            confidence_levels_ids = np.argwhere(
                np.array(plotting_matches.centroid_indices) == centroid_id
            ).reshape(-1)
            confidence_scores = [
                plotting_matches.confidence_levels[int(id)].value
                for id in confidence_levels_ids
            ]
            max_confidence_id = np.argwhere(
                np.array(confidence_scores) == max(confidence_scores)
            ).reshape(-1)
        return plotting_matches.confidence_levels[
            confidence_levels_ids[int(max_confidence_id)]
        ]

    def _assign_confidence_level_in_match(self, match_dictionary, centroid_id: int):
        structure_matcher_match = match_dictionary["structure_matcher"]
        ff_distance_match = (
            match_dictionary["structure_matcher"] <= self.fingerprint_distance_threshold
        )
        symmetry_match = match_dictionary["symmetry"]
        centroid_match = match_dictionary["centroid"] == centroid_id

        if structure_matcher_match:
            if centroid_match:
                return ConfidenceLevels.GOLD
            else:
                return ConfidenceLevels.HIGH
        else:
            if ff_distance_match and symmetry_match:
                return ConfidenceLevels.MEDIUM
            elif centroid_match and (ff_distance_match or symmetry_match):
                return ConfidenceLevels.MEDIUM
            else:
                return ConfidenceLevels.LOW

    def _compute_distance(
        self, structure_to_check: Structure, reference_structure: Structure
    ):
        if len(structure_to_check) == len(reference_structure):
            structure_to_check.sort()
            structure_to_check.sort()
            distance_to_known_structure = float(
                self.comparator._compare_structure(
                    AseAtomsAdaptor.get_atoms(structure_to_check),
                    AseAtomsAdaptor.get_atoms(reference_structure),
                )
            )
        else:
            distance_to_known_structure = 1
        return distance_to_known_structure

    def quick_view_structure(self, archive: Archive, individual_index: int):
        if not self.enable_structure_visualiser:
            return None
        else:
            structure = AseAtomsAdaptor.get_structure(archive.individuals[individual_index])
            self.structure_viewer.set_structure(structure)
            self.structure_viewer.show()

    def gif_centroid_over_time(
        self,
        experiment_directory_path: pathlib.Path,
        centroid_filepath: pathlib.Path,
        centroid_index: int,
        save_primitive: bool = False,
    ):
        list_of_files = [
            name
            for name in os.listdir(f"{experiment_directory_path}")
            if not os.path.isdir(name)
        ]
        list_of_archives = [
            filename
            for filename in list_of_files
            if ("archive_" in filename) and (".pkl" in filename)
        ]

        temp_dir = experiment_directory_path / "tempdir"
        temp_dir.mkdir(exist_ok=False)

        archive_ids = []
        plots = []
        for i, filename in enumerate(list_of_archives):
            if "relaxed_" in filename:
                continue
            else:
                archive_id = (
                    list_of_archives[i].lstrip("relaxed_archive_").rstrip(".pkl")
                )
                archive = Archive.from_archive(
                    pathlib.Path(experiment_directory_path / filename),
                    centroid_filepath=centroid_filepath,
                )
                archive_ids.append(archive_id)

                plot_name = self.save_structure_visualisations(
                    archive=archive,
                    structure_indices=[centroid_index],
                    directory_to_save=temp_dir,
                    file_tag=archive_id,
                    save_primitive=save_primitive,
                )
                plots.append(plot_name)

        frames = []
        sorting_ids = np.argsort(np.array(archive_ids, dtype=int))
        for id in sorting_ids:
            img = Image.open(str(temp_dir / plots[id]))
            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(img)
            # Add Text to an image
            I1.text((28, 36), archive_ids[id], fill=(255, 0, 0))
            # Display edited image
            img.show()
            # Save the edited image
            img.save(str(temp_dir / plots[id]))

            image = imageio.v2.imread(str(temp_dir / plots[id]))
            frames.append(image)

        imageio.mimsave(
            f"{experiment_directory_path}/structure_over_time_{centroid_index}.gif",  # output gif
            frames,
            duration=400,
        )
        for plot_name in plots:
            image_path = temp_dir / plot_name
            image_path.unlink()
        temp_dir.rmdir()

    def group_structures_by_symmetry(
        self,
        archive: Archive,
        experiment_directory_path: pathlib.Path,
    ):
        structures = archive.get_individuals_as_structures()
        groups = self.structure_matcher.group_structures(structures)
        ids_by_group = []
        for group in groups:
            id_in_group = []
            for el in group:
                match = [structures[i] == el for i in range(len(structures))]
                match_id = np.argwhere(np.array(match)).reshape(-1)
                id_in_group.append(archive.centroid_ids[match_id[0]])
            ids_by_group.append(id_in_group)

        with open(experiment_directory_path / "number_of_groups.json", "w") as file:
            json.dump({"n_groups": len(groups)}, file)

        color_indices = np.linspace(0, 1, len(ids_by_group))
        cmap = cm.get_cmap("rainbow")
        list_of_colors = []
        for color_id in color_indices:
            list_of_colors.append(cmap(color_id)[:3])
        return ids_by_group, list_of_colors

    @staticmethod
    def get_limits_from_centroid_path(centroid_path: pathlib.Path):
        filename = centroid_path.name.rstrip(".dat")
        limits_as_string = filename.split("band_gap")[1].split("shear_modulus")
        limits = [limit.split("_") for limit in limits_as_string]
        return (int(limits[0][1]), int(limits[1][1])), (
            int(limits[0][2]),
            int(limits[1][2]),
        )
