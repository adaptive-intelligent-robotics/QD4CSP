from argparse import ArgumentParser
from typing import List, Tuple

from qd4csp.reference_setup.reference_analyser import ReferenceAnalyser

def prepare_reference_data():
    elements_list = [["C"], ["Si", "O"], ["Si", "C"], ["Ti", "O"]]
    atoms_counts_list = [[24], [8, 16], [12, 12], [8, 16]]
    formulas = ["C", "SiO2", "SiC", "TiO2"]

    fitness_limits = None
    band_gap_limits = None
    shear_moduli_limits = None

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


def create_new_reference_material():
    parser = ArgumentParser()

    parser.add_argument(
        "-f",
        "--formula",
        help="Materials formula e.g. TiO2",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--element_list",
        help="List of elements in material e.g. [Ti, O]",
        type=List[str],
    )

    parser.add_argument(
        "-a",
        "--atoms_counts",
        help="List of counts of each atom e.g. [2, 4]",
        type=List[str],
    )

    parser.add_argument(
        "-fl",
        "--fitness_limits",
        help="Optional. Desired fitness limits for plottin in format [minimum_value, maximum_value].",
        type=Tuple[float, float],
        default=None,
    )

    parser.add_argument(
        "-b",
        "--band_gap_limits",
        help="Optional. Desired band gap limits for plotting in format [minimum_value, maximum_value].",
        type=Tuple[float, float],
        default=None,
    )

    parser.add_argument(
        "-s",
        "--shear_modulus_limits",
        help="Optional. Desired shear modulus limits for plotting in format [minimum_value, maximum_value].",
        type=Tuple[float, float],
        default=None,
    )

    args = parser.parse_args()

    ReferenceAnalyser.prepare_reference_data(
        formula=args.formula,
        elements_list=args.element_list,
        elements_counts_list=args.atom_counts,
        max_n_atoms_in_cell=sum(args.atom_counts),
        experimental_references_only=False,
        number_of_centroid_niches=200,
        fitness_limits=args.fitness_limits,
        band_gap_limits=args.band_gap_limits,
        shear_modulus_limits=args.shear_modulus_limits,
    )