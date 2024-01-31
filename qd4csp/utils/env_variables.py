import os
from pathlib import Path

EXPERIMENT_FOLDER = \
    Path(
        os.getenv(
            "EXPERIMENT_FOLDER",
            str(
                    (
                        Path(__file__).parent.parent.parent / "experiments"
                    ).absolute()),
        )
    )

CONFIGS_FOLDER = \
    Path(
        os.getenv(
            "CONFIGS_FOLDER",
            str(
                    (
                        Path(__file__).parent.parent.parent / "experiment_configs"
                    ).absolute()),
        )
    )

MP_REFERENCE_FOLDER =\
    Path(
        os.getenv(
            "MP_REFERENCE_FOLDER",
            str(
                (
                        Path(__file__).parent.parent.parent / "mp_reference_analysis"
                ).absolute()),
        )
    )

MP_API_KEY = os.getenv("MP_API_KEY")
