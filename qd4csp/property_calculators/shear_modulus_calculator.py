from typing import Optional, Tuple, List

from megnet.utils.models import load_model as megnet_load_model
from pymatgen.core import Structure

from qd4csp.utils.utils import normalise_between_0_and_1


class ShearModulusCalculator:
    def __init__(self, normalisation_values: Optional[Tuple[float, float]] = None):
        self.model_wrapper = megnet_load_model("logG_MP_2018")
        self.normalisation_values = normalisation_values

    def compute(self, structure: Structure):
        log_shear_modulus = self._compute_log_shear_modulus_no_gradients(structure)
        shear_modulus = 10**log_shear_modulus
        if self.normalisation_values is not None:
            shear_modulus = normalise_between_0_and_1(
                shear_modulus, self.normalisation_values
            )
        return shear_modulus

    def _compute_log_shear_modulus_no_gradients(self, structure: Structure):
        shear_modulus_log = self.model_wrapper.predict_structure(structure).ravel()[0]
        return shear_modulus_log

    def compute_no_grad_batch(self, list_of_structures: List[Structure]):
        return 10 ** self.model_wrapper.predict_structures(list_of_structures).ravel()
