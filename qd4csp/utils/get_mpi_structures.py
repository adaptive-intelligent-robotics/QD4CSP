from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from qd4csp.utils.env_variables import MP_API_KEY


def get_all_materials_with_formula(formula: str):
    """Gets all materials with a formula from the Materials Project.

    For more information on querying data from materials project visit:
    https://docs.materialsproject.org/downloading-data/using-the-api/querying-data
    """
    with MPRester(api_key=MP_API_KEY) as mpr:
        docs = mpr.summary.search(
            formula=formula,
            # band_gap=(0.5, 1.0),
            fields=[
                "material_id",
                "band_gap",
                "volume",
                "lattice",
                "formation_energy_per_atom",
                "energy_above_hull",
                "structure",
                "energy_per_atom",
                "uncorrected_energy_per_atom",
                "theoretical",
                "is_stable"
            ],
        )

    atoms = []
    for material in docs:
        new_atoms = AseAtomsAdaptor.get_atoms(material.structure)
        atoms.append(new_atoms)
    return docs, atoms
