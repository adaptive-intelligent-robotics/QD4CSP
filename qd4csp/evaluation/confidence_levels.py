from enum import Enum

from matplotlib import colors as mcolors


class ConfidenceLevels(int, Enum):
    GOLD = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NO_MATCH = 0

    @staticmethod
    def confidence_dictionary():
        return {
            ConfidenceLevels.GOLD: "Gold Standard",
            ConfidenceLevels.HIGH: "High",
            ConfidenceLevels.MEDIUM: "Medium",
            ConfidenceLevels.LOW: "Low",
            ConfidenceLevels.NO_MATCH: "No Match",
        }

    @staticmethod
    def energy_to_colour_dictionary():
        return {
            "no_match": mcolors.CSS4_COLORS["silver"],
            "energy_above_reference": mcolors.CSS4_COLORS["rosybrown"],
            "energy_below_reference": mcolors.CSS4_COLORS["mediumaquamarine"],
            ConfidenceLevels.GOLD.value: mcolors.TABLEAU_COLORS["tab:purple"],
        }

    @classmethod
    def get_string(cls, confidence_level: "ConfidenceLevel") -> str:
        return cls.confidence_dictionary()[confidence_level]

    @staticmethod
    def get_confidence_colour(confidence_level: "ConfidenceLevel") -> str:
        colour_dict = {
            ConfidenceLevels.GOLD: mcolors.TABLEAU_COLORS["tab:purple"],
            ConfidenceLevels.HIGH: mcolors.TABLEAU_COLORS["tab:green"],
            ConfidenceLevels.MEDIUM: mcolors.TABLEAU_COLORS["tab:orange"],
            ConfidenceLevels.LOW: mcolors.TABLEAU_COLORS["tab:red"],
            ConfidenceLevels.NO_MATCH: mcolors.TABLEAU_COLORS["tab:gray"],
        }

        return colour_dict[confidence_level]

    @classmethod
    def get_energy_comparison_colours(cls, energy_comparison: str) -> str:
        return cls.energy_to_colour_dictionary()[energy_comparison]
