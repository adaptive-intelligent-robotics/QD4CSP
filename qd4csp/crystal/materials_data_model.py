from enum import Enum


class BandGapEnum(int, Enum):
    PBE = 0
    GLLB_SC = 1
    MSE = 2
    SCAN = 3


class MaterialProperties(str, Enum):
    BAND_GAP = "band_gap"
    SHEAR_MODULUS = "shear_modulus"
    ENERGY = "energy"


class StartGenerators(str, Enum):
    RANDOM = "random"
    PYXTAL = "pyxtal"
