import json
import os
from pathlib import Path
from typing import List, Tuple
from typing import Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm

from qd4csp.evaluation.archive_statistics import ArchiveStatistics
from qd4csp.map_elites.archive import Archive
from qd4csp.utils.utils import get_all_files_at_location


class AverageStatisticsGenerator:
    def __init__(self, path_to_experiments: Path):
        self._path_to_all_experiments = path_to_experiments
        self.experiment_list = [
            name
            for name in os.listdir(f"{self._path_to_all_experiments}")
            if os.path.isdir(self._path_to_all_experiments / name)
               and (name not in  "all_plots")
        ]
        self.sub_experiment_list = self._list_all_individual_experiments()
        self.summary_plot_folder = self._path_to_all_experiments / "all_plots"
        self.summary_plot_folder.mkdir(exist_ok=True)
        self.non_experiment_directories = ["all_plots", "extra_experiments"]
        self.experiment_summary_filename = "ind_report_summary.json"

    def _list_all_individual_experiments(self) -> List[List[str]]:
        sub_experiments = []
        for experiment in self.experiment_list:
            path_to_experiment = self._path_to_all_experiments / experiment
            sub_experiments_by_exp = [
                name
                for name in os.listdir(f"{path_to_experiment}")
                if os.path.isdir(path_to_experiment / name)
            ]
            sub_experiments.append(sub_experiments_by_exp)
        return sub_experiments

    def plot_mean_statistics(
        self,
        folder_names: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        filename_tag: str = "",
        plot_individually: bool = True,
        filter_coverage_for_valid_solutions_only: bool = True,
        y_limits_dict: Optional[Dict[str, float]] = None,
    ):
        folders_to_plot = (
            folder_names if folder_names is not None else self.experiment_list
        )
        plot_labels = labels if labels is not None else self.experiment_list

        all_experiment_data = []

        for experiment in tqdm(folders_to_plot):
            i = np.argwhere(np.array(self.experiment_list) == experiment).reshape(-1)[0]
            data_for_1_experiment = []
            for j, experiment_name in enumerate(self.sub_experiment_list[i]):
                generation_data = self.compute_metrics_on_experiment(
                    path_to_subexperiment=self._path_to_all_experiments
                    / experiment
                    / experiment_name,
                    number_of_niches=200,
                    top_value=None,
                    filter_coverage_for_valid_solutions_only=filter_coverage_for_valid_solutions_only,
                )

                generation_data = generation_data.T
                data_for_1_experiment.append(generation_data)

            all_experiment_data.append(data_for_1_experiment)

        metric_names = [metric.value for metric in ArchiveStatistics]
        units = [ArchiveStatistics.unit(metric) for metric in ArchiveStatistics]

        all_processed_data = []

        for metric_id in tqdm(range(1, len(metric_names))):

            for i, experiment in enumerate(all_experiment_data):
                processed_data = []
                minimum_number_of_datapoints = min([len(el[0]) for el in experiment])
                for el in experiment:
                    processed_data.append(el[:, :minimum_number_of_datapoints])
                processed_data = np.array(processed_data)
                all_processed_data.append(processed_data)

                quartile_25 = np.percentile(processed_data, 25, axis=0)
                quartile_75 = np.percentile(processed_data, 75, axis=0)
                median = np.median(processed_data, axis=0)
                if len(all_experiment_data) > 1:
                    fig, ax = plt.subplots()
                    ax.plot(processed_data[0, 0], median[metric_id], label=plot_labels[i])
                    ax.fill_between(
                        processed_data[0, 0],
                        (quartile_25[metric_id]),
                        (quartile_75[metric_id]),
                        alpha=0.1,
                    )

                    ax.set_xlabel("Evaluation Count")
                    ax.set_ylabel(metric_names[metric_id] + units[metric_id])

                    if (
                            y_limits_dict is not None
                            and metric_names[metric_id] in y_limits_dict.keys()
                    ):
                        ax.set_ylim(y_limits_dict[metric_names[metric_id]])
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.2),
                        fontsize="small",
                        ncols=min(2, len(plot_labels)),
                    )
                    fig.tight_layout()
                    save_name = (
                        f"{filename_tag}_comparison_{metric_names[metric_id]}".replace(
                            " ", "_")
                            .lower()
                            .replace(".", "")
                            .replace("/", "")
                    )
                    fig.savefig(self.summary_plot_folder / f"{save_name}.png")
                    plt.close(fig)
                if plot_individually:
                    fig_ind, ax_ind = plt.subplots()

                    if metric_names[metric_id] == "Maximum Fitness":
                        ax_ind.plot(
                            processed_data[0, 0], median[metric_id],
                            label="MAP-Elites"
                        )
                        ax_ind.plot(processed_data[0, 0],
                                [9.407773971557617] * len(
                                    processed_data[0, 0]),
                                label="Ground State")
                        ax_ind.legend()
                        ax_ind.legend(
                            loc="upper center",
                            bbox_to_anchor=(0.5, -0.2),
                            fontsize="x-small",
                            ncols=2,
                        )

                    else:
                        ax_ind.plot(
                            processed_data[0, 0], median[metric_id],
                            label=plot_labels[i]
                        )
                    ax_ind.fill_between(
                        processed_data[0, 0],
                        (quartile_25[metric_id]),
                        (quartile_75[metric_id]),
                        alpha=0.1,
                    )

                    ax_ind.set_xlabel("Evaluation Count")
                    ax_ind.set_ylabel(metric_names[metric_id] + units[metric_id])
                    fig_ind.tight_layout()
                    save_name = (
                        f"{plot_labels[i]}_{metric_names[metric_id]}".replace(" ", "_")
                        .lower()
                        .replace(".", "")
                        .replace("/", "")
                    )
                    fig_ind.savefig(self.summary_plot_folder / f"{save_name}.png")
                    plt.close(fig_ind)

    def compute_metrics_on_experiment(
        self,
        path_to_subexperiment: Path,
        number_of_niches: int = 200,
        top_value: Optional[int] = None,
        filter_coverage_for_valid_solutions_only: bool = True,
    ):
        archive_strings = self.list_all_archives(path_to_subexperiment)
        all_data = []
        for i, archive_string in enumerate(archive_strings):
            archive = Archive.from_archive(path_to_subexperiment / archive_string)
            evaluation_number = int(archive_string.lstrip("archive_").rstrip(".pkl"))
            number_of_individuals = len(archive.fitnesses)
            fitness_metrics = archive.compute_fitness_metrics(top_value)
            coverage = archive.compute_coverage(
                number_of_niches, top_value, filter_coverage_for_valid_solutions_only
            )
            qd_score = archive.compute_qd_score(top_value)
            one_row = np.hstack(
                [
                    evaluation_number,
                    number_of_individuals,
                    fitness_metrics,
                    coverage,
                    qd_score,
                ]
            )
            all_data.append(one_row)

        return np.array(all_data)

    @staticmethod
    def list_all_archives(sub_experiment: Path) -> List[int]:
        list_of_files = [
            name for name in os.listdir(f"{sub_experiment}") if not os.path.isdir(name)
        ]
        list_of_archives = [
            filename
            for filename in list_of_files
            if ("archive_" in filename) and (".pkl" in filename)
        ]
        list_of_archive_ids = [
            int(filename.lstrip("archive_").rstrip(".pkl"))
            for filename in list_of_archives
        ]
        indices_to_sort = np.argsort(list_of_archive_ids)
        list_of_archives = np.take(list_of_archives, indices_to_sort)
        return list_of_archives

    def load_performance_summaries(self, experiment_directories: List[Path]) -> pd.DataFrame:
        summary_data_list = []

        for directory in experiment_directories:
            with open(directory / self.experiment_summary_filename, "r") as file:
                experiment_summary = json.load(file)
            experiment_summary["name"] = directory.name
            summary_data_list.append(experiment_summary)
        return pd.DataFrame(summary_data_list)

    def compute_performance_from_dataframe(
        self, experiment_dataframe: pd.DataFrame, directory_to_save: Path,
    ):
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

        with open(directory_to_save / "experiment_performance.json", "w") as file:
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

    def compute_average_match_performance(
            self, list_of_experiments: List[Path],
            directory_to_save: Path,
    ):
        performance_df = self.load_performance_summaries(list_of_experiments)
        self.compute_performance_from_dataframe(
            experiment_dataframe=performance_df,
            directory_to_save=directory_to_save,
        )

    def compute_average_match_performance_for_all_experiments(self):
        for i, experiment in enumerate(self.experiment_list):
            sub_experiments_list = [
                self._path_to_all_experiments / experiment / el
                for el in self.sub_experiment_list[i]
            ]
            directory_to_save = self._path_to_all_experiments / experiment
            self.compute_average_match_performance(
                list_of_experiments=sub_experiments_list,
                directory_to_save=directory_to_save,
            )
