import pathlib
from pathlib import Path
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

from qd4csp.crystal.materials_data_model import MaterialProperties


def initialise_kdt_and_centroids(
        bd_minimum_values: np.ndarray,
        bd_maximum_values: np.ndarray,
        number_of_niches: int,
        cvt_samples: int,
        experiment_directory_path: Path,
        behavioural_descriptors_names: List[MaterialProperties],
        cvt_use_cache: bool,
        formula: str,
) -> KDTree:
    # create the CVT
    assert len(bd_minimum_values) == len(bd_maximum_values)
    c = cvt(
        k=number_of_niches,
        number_of_bd_dimensions=len(bd_minimum_values),
        samples=cvt_samples,
        bd_minimum_values=bd_minimum_values,
        bd_maximum_values=bd_maximum_values,
        experiment_folder=experiment_directory_path,
        bd_names=behavioural_descriptors_names,
        cvt_use_cache=cvt_use_cache,
        formula=formula,
    )
    kdt = KDTree(c, leaf_size=30, metric="euclidean")
    write_centroids(
        c,
        experiment_folder=experiment_directory_path,
        bd_names=behavioural_descriptors_names,
        bd_minimum_values=bd_minimum_values,
        bd_maximum_values=bd_maximum_values,
        formula=formula,
    )
    del c
    return kdt


def __centroids_filename(
    k: int,
    dim: int,
    bd_names: List[MaterialProperties],
    bd_minimum_values: List[float],
    bd_maximum_values: List[float],
    formula: str,
) -> str:
    if formula == "TiO2" or formula is None:
        bd_tag = ""
    else:
        bd_tag = "_" + formula

    for i, bd_name in enumerate(bd_names):
        bd_tag += f"_{bd_name.value}_{bd_minimum_values[i]}_{bd_maximum_values[i]}"

    return "/centroids/centroids_" + str(k) + "_" + str(dim) + bd_tag + ".dat"


def write_centroids(
    centroids,
    experiment_folder,
    bd_names: List[MaterialProperties],
    bd_minimum_values: List[float],
    bd_maximum_values: List[float],
    formula: str,
):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(
        k, dim, bd_names, bd_minimum_values, bd_maximum_values, formula
    )
    file_path = Path(experiment_folder).parent
    with open(f"{file_path}{filename}", "w") as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + " ")
            f.write("\n")


def cvt(
    k,
    number_of_bd_dimensions,
    samples,
    bd_minimum_values,
    bd_maximum_values,
    experiment_folder,
    bd_names: List[MaterialProperties],
    cvt_use_cache=True,
    formula: str = "",
):
    # check if we have cached values
    fname = __centroids_filename(
        k,
        number_of_bd_dimensions,
        bd_names,
        bd_minimum_values,
        bd_maximum_values,
        formula=formula,
    )
    file_location = pathlib.Path(experiment_folder).parent
    if cvt_use_cache:
        if Path(f"{file_location}/{fname}").is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(f"{file_location}/{fname}")
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    bd_dim_1 = np.random.uniform(
        bd_minimum_values[0], bd_maximum_values[0], size=(samples, 1)
    )
    bd_dim_2 = np.random.uniform(
        bd_minimum_values[1], bd_maximum_values[1], size=(samples, 1)
    )
    x = np.hstack((bd_dim_1, bd_dim_2))
    if number_of_bd_dimensions == 3:
        bd_dim_3 = np.random.uniform(
            bd_minimum_values[1], bd_maximum_values[1], size=(samples, 1)
        )
        x = np.hstack((x, bd_dim_3))

    k_means = KMeans(
        init="k-means++", n_clusters=k, n_init=1, verbose=1
    )  # ,algorithm="full") ##  n_jobs=-1,
    k_means.fit(x)
    write_centroids(
        k_means.cluster_centers_,
        experiment_folder,
        bd_names,
        bd_minimum_values,
        bd_maximum_values,
        formula=formula,
    )

    return k_means.cluster_centers_


def make_hashable(array):
    return tuple(map(float, array))
