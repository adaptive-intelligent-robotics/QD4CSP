import numpy as np

from qd4csp.reference_setup.reference_analyser import ReferenceAnalyser

if __name__ == "__main__":
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
