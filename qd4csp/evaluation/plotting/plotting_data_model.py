from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Union, Tuple

import numpy as np
from matplotlib.axes import Axes

from qd4csp.evaluation.confidence_levels import ConfidenceLevels
from qd4csp.map_elites.archive import Archive


class PlotTypes(str, Enum):
    GROUPS = "groups"
    UNIQUE_MATCHES = "unique_matches"
    ALL_MATCHES = "all_matches"
    ENERGY = "energy"
    CVT = "cvt"

class PlottingMode(str, Enum):
    MP_REFERENCE_VIEW = "mp_reference_view"
    ARCHIVE_MATCHES_VIEW = "archive_matches_view"


@dataclass
class PlottingMatches:
    centroid_indices: List[int]
    mp_references: List[str]
    confidence_levels: List[ConfidenceLevels]
    euclidian_distances: List[float]
    descriptors: List[np.ndarray]
    energy_difference: List[float]
    plotting_mode: PlottingMode


@dataclass
class CVTPlottingData:
    plot_type: PlotTypes
    title: Optional[str]
    filename: str
    legend_label_dict: Optional[Dict[str, str]] = None
    legend_ax: Optional[Axes] = None
    archive: Optional[Archive] = None
    plotting_matches: Optional[PlottingMatches] = None
    target_centroid_ids: np.ndarray = None
    target_centroid_energies: Optional[np.ndarray] = None
    list_of_centroid_groups: Optional[List[List[int]]] = None
    list_of_colours: Optional[List[str]] = None
    reference_shear_moduli: Optional[List[float]] = None
    reference_band_gaps: Optional[List[float]] = None
    plotting_annotations: Optional[Union[List[str], np.ndarray]] = None,
    fitness_limits: Optional[Tuple[float, float]] = None

    @classmethod
    def for_structure_groups(
        cls,
        list_of_centroid_groups: List[List[int]],
        list_of_colours: List[str],
    ):
        return cls(
            plot_type=PlotTypes.GROUPS,
            title="Individuals Grouped by Similarity",
            filename="cvt_by_structure_similarity",
            list_of_centroid_groups=list_of_centroid_groups,
            list_of_colours=list_of_colours,
        )

    @classmethod
    def for_energy_comparison(
        cls,
        archive: Archive,
        plotting_matches: PlottingMatches,
        target_centroid_ids: np.ndarray,
        target_centroid_energies: np.ndarray,
    ):
        return cls(
            plot_type=PlotTypes.ENERGY,
            title="Energy Comparison",
            filename=f"cvt_energy_diff_matches_from_archive_{plotting_matches.plotting_mode.value}",
            archive=archive,
            plotting_matches=plotting_matches,
            target_centroid_energies=target_centroid_energies,
            target_centroid_ids=target_centroid_ids,
            legend_label_dict={
            ConfidenceLevels.GOLD.value: "Gold Standard",
            "energy_below_reference": "Energy Below Reference",
            "energy_above_reference": "Energy Above Reference",
            "no_match": "Not Accessed",
        },
        )

    @classmethod
    def for_reference_matching(
        cls,
        plotting_matches: PlottingMatches,
        target_centroid_ids: np.ndarray,
        reference_shear_moduli: Optional[List[float]],
        reference_band_gaps: Optional[List[float]],
    ):
        if plotting_matches.plotting_mode == PlottingMode.ARCHIVE_MATCHES_VIEW:
            plot_type = PlotTypes.ALL_MATCHES
            title = "Matches - Archive View"
        elif plotting_matches.plotting_mode == PlottingMode.MP_REFERENCE_VIEW:
            plot_type = PlotTypes.UNIQUE_MATCHES
            title = "Matches - Reference View"
            try:
                assert reference_shear_moduli is not None and reference_band_gaps is not None
            except AssertionError:
                raise ValueError("reference_shear_moduli and "
                                 "reference_band_gaps need to be provided when"
                                 " using {PlottingMode.MP_REFERENCE_VIEW")
        else:
            raise NotImplementedError("plotting mode must be MP_REFERENCE_VIEW"
                                      " or ARCHIVE_MATCHES_VIEW")

        return cls(
            plot_type=plot_type,
            title=title,
            filename=f"cvt_matches_from_archive_{plotting_matches.plotting_mode.value}",
            plotting_matches=plotting_matches,
            target_centroid_ids=target_centroid_ids,
            reference_shear_moduli=reference_shear_moduli,
            reference_band_gaps=reference_band_gaps,
            legend_label_dict=ConfidenceLevels.confidence_dictionary(),
        )

    @classmethod
    def for_cvt_plot(
        cls,
        archive: Archive,
        annotations: Optional[Union[List[str], np.ndarray]] = None,
        fitness_limits: Optional[Tuple[float, float]] = None
    ):
        return cls(
            plot_type=PlotTypes.CVT,
            title="MAP-Elites Grid",
            filename=f"cvt_plot_{archive.archive_number}",
            archive=archive,
            plotting_annotations=annotations,
            fitness_limits=tuple(fitness_limits),
        )
