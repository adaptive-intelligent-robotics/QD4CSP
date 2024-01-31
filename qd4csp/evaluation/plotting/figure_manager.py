from dataclasses import dataclass
from typing import Dict, Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qd4csp.evaluation.plotting.plotting_data_model import PlotTypes


@dataclass
class FigureManager:
    plot_to_axes_mapping: Dict[PlotTypes, Optional[Axes]]
    figure: Optional[Figure]

    @classmethod
    def create_empty(cls):
        plot_to_axes_mapping = {}
        for element in PlotTypes:
            plot_to_axes_mapping[element] = None

        return cls(plot_to_axes_mapping=plot_to_axes_mapping, figure=None)
