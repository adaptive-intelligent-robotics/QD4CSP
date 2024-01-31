import pathlib
import pickle
from typing import Tuple, List

import numpy as np


def normalise_between_0_and_1(
    values_to_normalise: np.ndarray, normalisation_limits: Tuple[float, float]
):
    return (values_to_normalise - normalisation_limits[0]) / (
        normalisation_limits[1] - normalisation_limits[0]
    )

def get_all_files_at_location(path_to_location: pathlib.Path) -> Tuple[List[str], List[str]]:
    all_filenames = [child for child in path_to_location.iterdir()]
    list_of_files = [filename.name for filename in all_filenames if filename.is_file()]
    list_of_directories = [filename.name for filename in all_filenames if filename.is_dir()]
    return list_of_files, list_of_directories


if __name__ == '__main__':
    list_of_files, list_of_directories = get_all_files_at_location(
         pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync/experiments/group_TiO2_benchmark")
    print()


def load_centroids(filename: str) -> np.ndarray:
    with open(filename, "r") as file:
        centroids = np.loadtxt(file)
    return centroids


def load_archive_from_pickle(filename: str):
    with open(filename, "rb") as file:
        archive = pickle.load(file)

    fitnesses = []
    centroids = []
    descriptors = []
    individuals = []
    for el in archive:
        fitnesses.append(el[0])
        centroids.append(list(el[1]))

        descriptors.append(list(el[2]))
        individuals.append(el[3])

    fitnesses = np.array(fitnesses)
    centroids = np.array(centroids)
    descriptors = np.array(descriptors)

    return fitnesses, centroids, descriptors, individuals
