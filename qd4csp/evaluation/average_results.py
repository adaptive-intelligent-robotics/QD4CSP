import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from qd4csp.utils.utils import get_all_files_at_location


class AverageExperiments:
    def __init__(self, experiment_location: Path):
        self.experiment_location = experiment_location
        self.non_experiment_directories = ["all_plots", "extra_experiments"]
        self.experiment_summary_filename = "ind_report_summary.json"

    def get_individual_experiment_names(self):
        _, directory_names = get_all_files_at_location(self.experiment_location)
        return [directory for directory in directory_names if directory not in self.non_experiment_directories]

    def load_performance_summaries(self, experiment_directories: List[str]) -> pd.DataFrame:

        summary_data_list = []
        for directory_name in experiment_directories:
            directory = self.experiment_location / directory_name
            with open(directory / self.experiment_summary_filename, "r") as file:
                experiment_summary = json.load(file)
            experiment_summary["name"] = directory_name
            summary_data_list.append(experiment_summary)

        return pd.DataFrame(summary_data_list)

    def compute_performance_from_dataframe(self, experiment_dataframe: pd.DataFrame):
        performance_dictionary = {}

        # is ground state found
        ground_state_list = list(
            experiment_dataframe[
                ['ground_state_match', 'fooled_ground_state_match']
            ].itertuples(index=False, name=None)
        )

        gold_matches = np.array(
            ["Gold Standard" in match_tuple
             for match_tuple in ground_state_list]
        )

        high_matches = np.array(
            ["High" in match_tuple for match_tuple in ground_state_list]
        )
        proportion_ground_state = \
            np.logical_or(high_matches, gold_matches).sum() \
            / len(ground_state_list)

        performance_dictionary["proportion_ground_state"] = \
            proportion_ground_state

        performance_dictionary["number_ground_state"] = \
            float(np.logical_or(high_matches, gold_matches).sum())

        # unique reference matches
        performance_dictionary["average_unique_reference_matches_mean"] = \
            experiment_dataframe["unique_reference_matches"].mean()

        performance_dictionary["average_unique_reference_matches_std"] = \
        experiment_dataframe["unique_reference_matches"].std()

        performance_dictionary["average_gold_and_high_matches_mean"] = \
            (
                experiment_dataframe["number_gold_unique"] +
                experiment_dataframe["number_high_unique"]
             ).mean()

        performance_dictionary["average_gold_and_high_matches_std"] = \
            (
                experiment_dataframe["number_gold_unique"] +
                experiment_dataframe["number_high_unique"]
             ).std()

        # portion of matches that are unique
        proportion_unique_matches = \
            (experiment_dataframe["unique_reference_matches"]
             / experiment_dataframe["total_matches"]).to_numpy()

        performance_dictionary["proportion_of_unique_matches_mean"] = \
            proportion_unique_matches.mean()

        performance_dictionary["proportion_of_unique_matches_std"] = \
            proportion_unique_matches.std()

        with open(self.experiment_location / "experiment_performance.json", "w") as file:
            json.dump(performance_dictionary, file)

    @staticmethod
    def rank_experiments(experiment_dataframe: pd.DataFrame):
        experiment_dataframe["number_medium_and_low_sum_unique"] = \
            experiment_dataframe["number_low_unique"] + \
            experiment_dataframe["number_medium_unique"]
        sorted_experiment_dataframe = experiment_dataframe.sort_values(
            by=[
                "number_gold_unique", "number_high_unique",
                "number_medium_and_low_sum_unique",
            ],
            ascending=[False, False, False],
        )

        return sorted_experiment_dataframe

    def select_random_experiment_for_paper(
        self,
        number_of_experiments_to_discard: int,
        number_of_experiments: int,
    ):
        return np.random.randint(
            low=number_of_experiments_to_discard,
            high=number_of_experiments - number_of_experiments_to_discard,
        )

    def join_structure_pngs_into_one_figure(
        self,
        image_paths: List[Path],
        filename: str,
        path_to_save: Path,
        number_of_rows: int = 2,
    ):
        margin = 30
        im = Image.open(image_paths[0])
        full_image_width, full_image_height = im.size
        scale = 1
        number_of_images = len(image_paths)
        images_per_row = number_of_images // number_of_rows + number_of_images % 2
        image_width = int(full_image_width / images_per_row * scale)
        image_height = int(full_image_height / images_per_row*scale)
        new_image = Image.new(
            "RGB",
            (image_width * images_per_row,
             (image_height + margin) * number_of_rows),
            "white",
        )
        for i, image in enumerate(image_paths):
            match, confidence = self._parse_structure_name_to_data(image.name)
            column = i % images_per_row
            row_id = (i // images_per_row)
            row = row_id * (image_height)
            img = Image.open(image)
            img_size = img.resize((image_width, image_height))
            image_draw = ImageDraw.Draw(new_image)
            new_image.paste(
                img_size,
                (image_width * column,
                 row + (30 * (row_id + 1)))
            )
            image_draw.text(
                (image_width * column, row + row_id * margin),
                f"{match} - {confidence}",
                font_size=20,
                fill=(0, 0, 0),
            )
        new_image.save(path_to_save / filename)

    def get_list_of_structure_images(
        self,
        experiment_name: str
    ) -> Tuple[List[Path], List[Path]]:
        image_paths = [
            el for el in
            (self.experiment_location / experiment_name).iterdir()
            if ("ind_" in el.name) and ".png"
        ]
        primitive_structures = [
            el for el in image_paths if
            ("_primitive" in el.name) and (".png" in el.name) and ("cif" in el.name)
        ]
        full_structures = [
            el for el in image_paths if
            (not "_primitive" in el.name) and (".png" in el.name)
        ]
        return primitive_structures, full_structures

    def _parse_structure_name_to_data(self, structure_filename: str):
        structure_information = structure_filename.split("_")
        if "cif" in structure_filename:
            confidence = structure_information[4]
            match = structure_information[3]
        else:
            confidence = structure_information[3]
            match = structure_information[2]

        if confidence == "goldstandard":
            confidence = "Gold Standard"
        elif confidence == "high":
            confidence = "High"
        elif confidence == "medium":
            confidence = "Medium"
        elif confidence == "low":
            confidence = "Low"
        return match, confidence

if __name__ == '__main__':
    averager = AverageExperiments(Path(
        __file__).parent.parent.parent / f".experiment.nosync/experiments/TiO2/group_TiO2_benchmark")
    experiment_name = "20231030_16_51_TiO2_benchmark_1_3"
    primitive_structure_paths, full_structure_paths  = averager.get_list_of_structure_images(
                experiment_name=experiment_name,
            )
    sorting_dict = {"goldstandard": 4, "high": 3, "medium": 2, "low": 1}
    primitive_structure_paths.sort(
        key=lambda x: sorting_dict[str(x.name).split("_")[4]], reverse=True)
    averager.join_structure_pngs_into_one_figure(
        image_paths=primitive_structure_paths,
        filename="tio2_joined_primitive_crystals.png",
        path_to_save=averager.experiment_location / experiment_name,
        number_of_rows=2,
    )
