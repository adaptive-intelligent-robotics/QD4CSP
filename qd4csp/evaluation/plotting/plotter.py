from typing import Dict, Optional, List, Tuple, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qd4csp.evaluation.plotting.plotting_data_model import PlotTypes, \
    CVTPlottingData, PlottingMode, PlottingMatches
from qd4csp.evaluation.confidence_levels import ConfidenceLevels
from qd4csp.map_elites.archive import Archive
from qd4csp.evaluation.plotting.utils import get_voronoi_finite_polygons_2d

plt.rcParams["savefig.dpi"] = 300

class CVTPlotting:
    def __init__(
        self,
        centroids: np.ndarray,
        axes_minimum_values: np.ndarray, # equivalent to minval
        axes_maximum_values: np.ndarray, # euivalent to maxval
        target_centroids: Optional[np.ndarray] = None,
        axis_labels: List[str] = ["Band Gap, eV", "Shear Modulus, GPa"],
        directory_string: Optional[str] = None,
        overload_x_axis_limits: Optional[Tuple[float, float]] = None,
        overload_y_axis_limits: Optional[Tuple[float, float]] = None,
    ):
        self.centroids = centroids
        self.axes_minimum_values = axes_minimum_values
        self.axes_maximum_values = axes_maximum_values
        self.target_centroids = target_centroids
        self.axis_labels = axis_labels
        self.directory_string = directory_string
        self.overload_x_axis_limits = overload_x_axis_limits
        self.overload_y_axis_limits = overload_y_axis_limits
        self.regions, self.vertices = get_voronoi_finite_polygons_2d(centroids)

    def _set_axes_limits(self, ax: Axes):
        if len(np.array(self.axes_minimum_values).shape) == 0 and len(np.array(self.axes_maximum_values).shape) == 0:
            ax.set_xlim(self.axes_minimum_values, self.axes_maximum_values)
            ax.set_ylim(self.axes_minimum_values, self.axes_maximum_values)
        else:
            ax.set_xlim(self.axes_minimum_values[0], self.axes_maximum_values[0])
            ax.set_ylim(self.axes_minimum_values[1], self.axes_maximum_values[1])

        ax.set(adjustable="box", aspect="equal")
        return ax

    def _fill_plot_with_contours(self, ax: Axes) -> Axes:
        for i, region in enumerate(self.regions):
            polygon = self.vertices[region]
            ax.fill(
                *zip(*polygon),
                alpha=0.05, edgecolor="black", facecolor="white", lw=1
            )
        return ax

    def _add_contours_for_target_centroids(
        self, ax: Axes, plot_type: PlotTypes,
    ) -> Axes:
        if plot_type == PlotTypes.CVT:
            color, line_width = mcolors.CSS4_COLORS["salmon"], 1
        else:
            color, line_width = "gray", 2

        for i, region in enumerate(self.regions):
            polygon = self.vertices[region]
            if self.target_centroids is not None:
                if self.centroids[i] in np.array(self.target_centroids):
                    ax.fill(*zip(*polygon),
                            edgecolor=color, facecolor="none",
                            lw=line_width,
                            )
        return ax

    def plot(
        self,
        ax: Optional[Axes],
        fig: Optional[Figure],
        plotting_metadata: CVTPlottingData,
        annotate: bool = False,
    ) -> Axes:
        fig, ax = (fig, ax) \
            if ax is not None else \
            plt.subplots(facecolor="white", edgecolor="white")

        ax = self._set_axes_limits(ax=ax)
        ax = self._fill_plot_with_contours(ax=ax)
        ax = self._fill_plot(
            ax=ax, plotting_metadata=plotting_metadata, annotate=annotate
        )

        if plotting_metadata.legend_label_dict is not None:
            ax = self.legend_without_duplicate_labels(
                ax=ax,
                sorting_match_list=list(plotting_metadata.legend_label_dict.values())
                )

        if self.target_centroids is not None:
            ax = self._add_contours_for_target_centroids(
                ax=ax, plot_type=plotting_metadata.plot_type,
            )

        ax = self._update_plot_annotations(
            title=plotting_metadata.title,
            ax=ax
        )
        if fig is not None:
            self.display_figure(filename=plotting_metadata.filename, fig=fig)

        return ax

    def _fill_plot(
        self,
        ax: Axes,
        plotting_metadata: CVTPlottingData,
        annotate: bool = False,
    ) -> Axes:
        if plotting_metadata.plot_type == PlotTypes.GROUPS:
            ax = self._plot_2d_groups(
                ax=ax, annotate=annotate,
                list_of_centroid_groups=plotting_metadata.list_of_centroid_groups,
                list_of_colors=plotting_metadata.list_of_colours,
            )
        elif plotting_metadata.plot_type == PlotTypes.ENERGY:
            ax = self._plot_energy_comparison(
                ax=ax, annotate=annotate,
                archive=plotting_metadata.archive,
                plotting_matches=plotting_metadata.plotting_matches,
                target_centroid_ids=plotting_metadata.target_centroid_ids,
                target_centroid_energies=plotting_metadata.target_centroid_energies,
                legend_labels=plotting_metadata.legend_label_dict,
            )

        elif plotting_metadata.plot_type == PlotTypes.ALL_MATCHES or \
                plotting_metadata.plot_type == PlotTypes.UNIQUE_MATCHES:
            ax = self._plot_matches_mapped_to_references(
                ax=ax,
                annotate=annotate,
                plotting_matches=plotting_metadata.plotting_matches,
                reference_shear_moduli=plotting_metadata.reference_shear_moduli,
                reference_band_gaps=plotting_metadata.reference_band_gaps,
            )
        elif plotting_metadata.plot_type == PlotTypes.CVT:
            ax = self._plot_cvt(
                ax=ax,
                annotate=annotate,
                archive=plotting_metadata.archive,
                fitness_limits=plotting_metadata.fitness_limits,
            )

        return ax

    def _update_plot_annotations(self, title: str, ax: Axes):
        np.set_printoptions(2)
        ax.set_xlabel(f"{self.axis_labels[0]}")
        ax.set_ylabel(f"{self.axis_labels[1]}")

        if self.overload_x_axis_limits is not None and self.overload_y_axis_limits is not None:
            np.set_printoptions(2)
            x_tick_labels = np.linspace(self.overload_x_axis_limits[0], self.overload_x_axis_limits[1], 6)
            x = np.linspace(0, 1, 6)
            ax.set_xticks(x, labels=["{:.1f}".format(value) for value in x_tick_labels])

            y_tick_labels = np.linspace(self.overload_y_axis_limits[0], self.overload_y_axis_limits[1], 6)
            y = np.linspace(0, 1, 6)
            ax.set_yticks(y, labels=["{:.1f}".format(value) for value in x_tick_labels])
            # ax.set_xticklabels([np.around(el, 1) for el in x_tick_labels])
            # ax.set_yticklabels([np.around(el, 1) for el in y_tick_labels])

        if title is not None:
            ax.set_title(title)
        ax.set_aspect("equal")
        return ax

    def display_figure(self, filename: str, fig: Figure):
        if self.directory_string is None:
            fig.show()
        elif self.directory_string is not None:
            fig.savefig(
                f"{self.directory_string}/{filename}.png",
                format="png",
                bbox_inches="tight",
            )
        else:
            pass

    def legend_without_duplicate_labels(
        self, ax: Axes, sorting_match_list: Optional[List[str]] = None,
    ) -> Axes:
        """Creates legend without duplicate labels."""

        try:
            handles, labels = ax.get_legend_handles_labels()
            unique = [
                (h, l)
                for i, (h, l) in enumerate(zip(handles, labels))
                if l not in labels[:i]
            ]
            if sorting_match_list is None:
                sorting_match_list = \
                    np.array(
                        [
                            ConfidenceLevels.get_string(el) for el in list(ConfidenceLevels)
                        ]
                )
            unique = sorted(
                unique,
                key=lambda x: np.argwhere(np.array(sorting_match_list) == x[1]).reshape(-1)[0],
            )
            ax.legend(
                *zip(*unique),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                fontsize="small",
                ncols=2,
            )
        except Exception as e:
            print("Error generating legend.")
            pass

        return ax

    def _plot_2d_groups(
        self,
        ax: Axes,
        annotate: bool,
        list_of_centroid_groups: List[List[int]],
        list_of_colors: List[str],
    ) -> Axes:
        for group_id, group in enumerate(list_of_centroid_groups):
            for idx in group:
                region = self.regions[idx]
                polygon = self.vertices[region]
                ax.fill(*zip(*polygon), alpha=0.8, color=list_of_colors[group_id])
                if annotate:
                    ax.annotate(
                        str(group_id),
                        (self.centroids[idx, 0], self.centroids[idx, 1]),
                        fontsize=4
                    )
        return ax

    def _plot_energy_comparison(
        self,
        ax: Axes,
        annotate: bool,
        archive: Archive,
        plotting_matches: PlottingMatches,
        target_centroid_ids: np.ndarray,
        target_centroid_energies: np.ndarray,
        legend_labels: Dict[str, str],
    ) -> Axes:
        duplicate_centroid_indices = []
        if len(np.unique(np.array(plotting_matches.centroid_indices))) != len(
            plotting_matches.centroid_indices
        ):
            unique, counts = np.unique(
                plotting_matches.centroid_indices, return_counts=True
            )
            duplicate_centroid_indices = np.take(
                unique, np.argwhere(counts != 1)
            ).reshape(-1)

        for i, region in enumerate(self.regions):
            polygon = self.vertices[region]
            if i in target_centroid_ids:
                target_centroid_energy = target_centroid_energies[
                    np.argwhere(target_centroid_ids == i)
                ].reshape(-1)[0]
                if i in archive.centroid_ids:
                    centroid_energy = archive.fitnesses[
                        np.argwhere(np.array(archive.centroid_ids) == i)
                    ].reshape(-1)[0]
                    if centroid_energy < target_centroid_energy:
                        ax.fill(
                            *zip(*polygon),
                            facecolor=ConfidenceLevels.get_energy_comparison_colours("energy_below_reference"),
                            label= legend_labels["energy_below_reference"],
                        )
                    elif (
                        i in archive.centroid_ids
                        and centroid_energy >= target_centroid_energy
                    ):
                        ax.fill(
                            *zip(*polygon),
                            facecolor=ConfidenceLevels.get_energy_comparison_colours("energy_above_reference"),
                            label=legend_labels["energy_above_reference"],
                        )
                else:
                    # continue
                    ax.fill(
                        *zip(*polygon),
                        facecolor=ConfidenceLevels.get_energy_comparison_colours("no_match"),
                        label=legend_labels["no_match"],
                    )

        for list_index, centroid_index in enumerate(plotting_matches.centroid_indices):
            region = self.regions[centroid_index]
            polygon = self.vertices[region]

            if centroid_index in duplicate_centroid_indices:
                confidence_levels_ids = np.argwhere(
                    np.array(plotting_matches.centroid_indices) == centroid_index
                ).reshape(-1)
                confidence_scores = [
                    plotting_matches.confidence_levels[int(id)].value
                    for id in confidence_levels_ids
                ]
                max_confidence_id = np.argwhere(
                    np.array(confidence_scores) == max(confidence_scores)
                ).reshape(-1)
                confidence_level = plotting_matches.confidence_levels[
                    int(max_confidence_id[0])
                ]
            else:
                confidence_level = plotting_matches.confidence_levels[list_index]

            if confidence_level == ConfidenceLevels.GOLD:
                ax.fill(
                    *zip(*polygon),
                    alpha=0.8,
                    color=ConfidenceLevels.get_energy_comparison_colours(ConfidenceLevels.GOLD.value),
                    label=legend_labels[ConfidenceLevels.GOLD.value],
                )
            if annotate:
                ax.annotate(
                    plotting_matches.mp_references[list_index],
                    (self.centroids[centroid_index, 0], self.centroids[centroid_index, 1]),
                    fontsize=4,
                )

        return ax

    def _plot_matches_mapped_to_references(
        self,
        ax: Axes,
        annotate: bool,
        plotting_matches: PlottingMatches,
        reference_shear_moduli: List[float],
        reference_band_gaps: List[float],
    ):
        duplicate_centroid_indices = []
        if len(np.unique(np.array(plotting_matches.centroid_indices))) != len(
            plotting_matches.centroid_indices
        ):
            unique, counts = np.unique(
                plotting_matches.centroid_indices, return_counts=True
            )
            duplicate_centroid_indices = np.take(
                unique, np.argwhere(counts != 1)
            ).reshape(-1)

        for i, region in enumerate(self.regions):
            polygon = self.vertices[region]
            if (
                (i in self.centroids)
                and (i not in plotting_matches.centroid_indices)
                and plotting_matches.plotting_mode == PlottingMode.ARCHIVE_MATCHES_VIEW
            ):
                ax.fill(
                    *zip(*polygon),
                    facecolor=ConfidenceLevels.get_confidence_colour(ConfidenceLevels.NO_MATCH),
                    alpha=0.3,
                    label=ConfidenceLevels.get_string(ConfidenceLevels.NO_MATCH),
                )

        for list_index, centroid_index in enumerate(plotting_matches.centroid_indices):
            region = self.regions[centroid_index]
            polygon = self.vertices[region]

            if centroid_index in duplicate_centroid_indices:
                confidence_levels_ids = np.argwhere(
                    np.array(plotting_matches.centroid_indices) == centroid_index
                ).reshape(-1)
                confidence_scores = [
                    plotting_matches.confidence_levels[int(id)].value
                    for id in confidence_levels_ids
                ]
                max_confidence_id = np.argwhere(
                    np.array(confidence_scores) == max(confidence_scores)
                ).reshape(-1)

                colour = ConfidenceLevels.get_confidence_colour(
                    plotting_matches.confidence_levels[
                        confidence_levels_ids[int(max_confidence_id[0])]
                    ]
                )
                confidence_level = plotting_matches.confidence_levels[
                    confidence_levels_ids[int(max_confidence_id[0])]
                ]
            else:
                colour = ConfidenceLevels.get_confidence_colour(plotting_matches.confidence_levels[list_index])
                confidence_level = plotting_matches.confidence_levels[list_index]

            ax.fill(
                *zip(*polygon),
                alpha=0.8,
                color=colour,
                label=ConfidenceLevels.get_string(confidence_level),
            )
            if annotate:
                ax.annotate(
                    plotting_matches.mp_references[list_index],
                    (self.centroids[centroid_index, 0], self.centroids[centroid_index, 1]),
                    fontsize=4,
                )

        if plotting_matches.plotting_mode == PlottingMode.MP_REFERENCE_VIEW:
            descriptor_matches = np.array(plotting_matches.descriptors)
            ax.scatter(
                descriptor_matches[:, 0], descriptor_matches[:, 1], color="b", s=5
            )
            ax.scatter(reference_band_gaps, reference_shear_moduli, color="b", s=5)
            for match_id in range(len(descriptor_matches)):
                ax.plot(
                    [descriptor_matches[match_id][0], reference_band_gaps[match_id]],
                    [descriptor_matches[match_id][1], reference_shear_moduli[match_id]],
                    linestyle="--",
                    color="b",
                )
        return ax

    def _plot_cvt(
        self,
        ax: Axes,
        archive: Archive,
        fitness_limits: Optional[Tuple[float, float]],
        annotate: bool,
    ) -> Axes:
        fitnesses, descriptors, annotations = \
            archive.convert_fitness_and_descriptors_to_plotting_format(
            self.centroids
        )

        grid_empty = fitnesses == -np.inf
        my_cmap = cm.viridis

        vmin, vmax = fitness_limits if isinstance(fitness_limits, tuple) else (None, None)

        if vmin is None:
            vmin = float(np.min(fitnesses[~grid_empty]))
        if vmax is None:
            vmax = float(np.max(fitnesses[~grid_empty]))

        norm = Normalize(vmin=vmin, vmax=vmax)

        # fill the plot with the colors5t
        for idx, fitness in enumerate(fitnesses):
            if fitness > -np.inf:
                region = self.regions[idx]
                polygon = self.vertices[region]
                ax.fill(*zip(*polygon), alpha=0.8,
                        color=my_cmap(norm(fitness)))

        np.set_printoptions(2)
        if descriptors is not None:
            descriptors = descriptors[~grid_empty]
            ax.scatter(
                descriptors[:, 0],
                descriptors[:, 1],
                c="black",
                s=1,
                zorder=0,
            )
            for i in range(len(fitnesses)):
                if annotate:
                    if annotations is None:
                        annotations = np.around(fitnesses, decimals=3)
                    if isinstance(annotations[i], str) \
                            and annotations[i] != "-inf":
                        ax.annotate(annotations[i],
                                    (self.centroids[i, 0], self.centroids[i, 1]))
                    elif isinstance(annotations[i], float) \
                            and annotations[i] != -np.inf:
                        ax.annotate(
                            annotations[i], (self.centroids[i, 0], self.centroids[i, 1]),
                            fontsize=4
                        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap),
                            cax=cax)
        cbar.ax.tick_params(labelsize=mpl.rcParams["font.size"])
        return ax
