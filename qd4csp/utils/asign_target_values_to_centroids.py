import pickle
from typing import Optional, List, Tuple

import numpy as np
from sklearn.neighbors import KDTree

from qd4csp.map_elites.cvt_centroids.initialise import make_hashable
from qd4csp.utils.get_mpi_structures import get_all_materials_with_formula
from qd4csp.utils.utils import normalise_between_0_and_1


def reassign_data_from_pkl_to_new_centroids(
    centroids_file: str,
    target_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    filter_for_number_of_atoms: Optional[int],
    normalise_bd_values: Optional[Tuple[List[float], List[float]]],
):
    with open(centroids_file, "r") as f:
        c = np.loadtxt(f)
    kdt = KDTree(c, leaf_size=30, metric="euclidean")

    fitnesses, _, descriptors, individuals = target_data
    if normalise_bd_values is not None:
        descriptors[:, 0] = normalise_between_0_and_1(
            descriptors[:, 0], (normalise_bd_values[0][0], normalise_bd_values[1][0])
        )
        descriptors[:, 1] = normalise_between_0_and_1(
            descriptors[:, 1], (normalise_bd_values[0][1], normalise_bd_values[1][1])
        )

    fitnesses_to_enumerate = []
    band_gaps = []
    shear_moduli = []

    if filter_for_number_of_atoms is not None:
        for i, atoms in enumerate(individuals):
            atom_positions = (
                atoms["positions"] if isinstance(atoms, dict) else atoms.get_positions()
            )
            if len(atom_positions) <= filter_for_number_of_atoms:
                fitnesses_to_enumerate.append(fitnesses[i])
                band_gaps.append(descriptors[i][0])
                shear_moduli.append(descriptors[i][1])
    else:
        for i, atoms in enumerate(fitnesses):
            fitnesses_to_enumerate.append(fitnesses[i])
            band_gaps.append(descriptors[i][0])
            shear_moduli.append(descriptors[i][1])
    new_centroids = []
    for i in range(len(fitnesses_to_enumerate)):
        niche_index = kdt.query([(band_gaps[i], shear_moduli[i])], k=1)[1][0][0]
        niche = kdt.data[niche_index]
        n = make_hashable(niche)
        new_centroids.append(n)

    return new_centroids
