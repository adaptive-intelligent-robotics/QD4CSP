import os
from typing import Optional, Tuple, List, Dict

import imageio
import matplotlib as mpl

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi


def get_voronoi_finite_polygons_2d(
    centroids: np.ndarray, radius: Optional[float] = None
) -> Tuple[List, np.ndarray]:
    """COPIED FROM QDAX
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions."""
    voronoi_diagram = Voronoi(centroids)
    if voronoi_diagram.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = voronoi_diagram.vertices.tolist()

    center = voronoi_diagram.points.mean(axis=0)
    if radius is None:
        radius = voronoi_diagram.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges: Dict[np.ndarray, np.ndarray] = {}
    for (p1, p2), (v1, v2) in zip(
        voronoi_diagram.ridge_points, voronoi_diagram.ridge_vertices
    ):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(voronoi_diagram.point_region):
        vertices = voronoi_diagram.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = voronoi_diagram.points[p2] - voronoi_diagram.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = voronoi_diagram.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = voronoi_diagram.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_all_statistics_from_file(filename: str, save_location: Optional[str]):
    with open(filename, "r") as file:
        generation_data = np.loadtxt(file)

    generation_data = generation_data.T

    number_of_metrics = len(generation_data)
    if number_of_metrics == 7:
        metric_names = [
            "Evaluation number",
            "Archive size",
            "Maximum Fitness",
            "Mean Fitness",
            "Median Fitness",
            "Fitness 5th Percentile",
            "Fitness 95th Percentile",
        ]
    elif number_of_metrics == 9:
        metric_names = [
            "Evaluation number",
            "Archive size",
            "Maximum Fitness",
            "Mean Fitness",
            "Median Fitness",
            "Fitness 5th Percentile",
            "Fitness 95th Percentile",
            "Coverage",
            "QD score",
        ]
    else:
        raise ValueError("unknown metric present in log file, check, number of columns")

    for metric_id in range(1, number_of_metrics):
        fig, ax = plt.subplots(figsize=[3.5, 2.625])
        ax.plot(generation_data[0], generation_data[metric_id])
        ax.set_ylabel(metric_names[metric_id])
        ax.set_xlabel("Evaluation Count")
        if save_location is None or save_location == "":
            plt.show()
        else:
            file_tag = metric_names[metric_id].replace(" ", "")
            plt.tight_layout()
            plt.savefig(f"{save_location}stats_{file_tag}")


def plot_gif(experiment_directory_path: str):
    plot_list = [
        name
        for name in os.listdir(f"{experiment_directory_path}")
        if not os.path.isdir(name) and "cvt_plot_" in name and ".png" in name
    ]
    sorted_plot_list = sorted(
        plot_list, key=lambda x: int(x.lstrip("cvt_plot_").rstrip(".png"))
    )

    frames = []
    for plot_name in sorted_plot_list:
        image = imageio.v2.imread(f"{experiment_directory_path}/{plot_name}")
        frames.append(image)

    imageio.mimsave(
        f"{experiment_directory_path}/cvt_plot_gif.gif",  # output gif
        frames,
    )  # array of input frames)
