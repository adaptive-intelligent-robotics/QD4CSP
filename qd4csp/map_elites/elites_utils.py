# | This file is based on the implementation map-elites implementation pymap_elites repo by resibots team https://github.com/resibots/pymap_elites
# | Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
# | Eloise Dalin , eloise.dalin@inria.fr
# | Pierre Desreumaux , pierre.desreumaux@inria.fr
# | **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
# | mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#

import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from ase import Atoms

from qd4csp.map_elites.cvt_centroids.initialise import make_hashable
from qd4csp.utils.env_variables import EXPERIMENT_FOLDER


class Species:
    def __init__(
        self,
        x,
        desc,
        fitness,
        centroid=None,
    ):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid


# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def save_archive(archive, gen, directory_path):
    storage = []
    for k in archive.values():
        one_individual = [k.fitness, k.centroid, k.desc, k.x]
        storage.append(one_individual)

    filename_pkl = str(directory_path) + "/archive_" + str(gen) + ".pkl"
    with open(filename_pkl, "wb") as f:
        pickle.dump(storage, f)


def add_to_archive(
    s: Species, centroid: np.ndarray, archive: Dict[str, Species], kdt
) -> Tuple[bool, int]:
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = make_hashable(niche)
    s.centroid = n
    info = s.x.info if isinstance(s.x, Atoms) else s.x["info"]
    if "data" in info:
        parent_id = info["data"]["parents"]
    else:
        parent_id = [None]
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return True, parent_id
        return False, parent_id
    else:
        archive[n] = s
        return True, parent_id


def make_experiment_folder(directory_name: str):
    new_path = EXPERIMENT_FOLDER / directory_name
    new_path.mkdir(exist_ok=True)
    return new_path


def make_current_time_string(with_time: bool = True):
    today = date.today().strftime("%Y%m%d")
    time_now = datetime.now().strftime("%H_%M") if with_time else ""
    return f"{today}_{time_now}"
