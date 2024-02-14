from enum import Enum


class ArchiveStatistics(str, Enum):
    EVALUATION_NUMBER = "Evaluation number"
    ARCHIVE_SIZE = "Archive size"
    MAXIMUM_FITNESS = "Maximum Fitness"
    MEAN_FITNESS = "Mean Fitness"
    MEDIAN_FITNESS = "Median Fitness"
    FITNESS_5_PERCENTILE = "Fitness 5th Percentile"
    FITNESS_95_PERCENTILE = "Fitness 95th Percentile"
    COVERAGE = "Coverage"
    QD_SCORE = "QD score"

    @classmethod
    def unit(cls, metric: "ArchiveStatistics") -> str:
        unit_dict = {
            ArchiveStatistics.EVALUATION_NUMBER: "",
            ArchiveStatistics.ARCHIVE_SIZE: "",
            ArchiveStatistics.MAXIMUM_FITNESS: ", eV/atom",
            ArchiveStatistics.MEAN_FITNESS: ", eV/atom",
            ArchiveStatistics.MEDIAN_FITNESS: ", eV/atom",
            ArchiveStatistics.FITNESS_5_PERCENTILE: ", eV/atom",
            ArchiveStatistics.FITNESS_95_PERCENTILE: ", eV/atom",
            ArchiveStatistics.COVERAGE: ", \%",
            ArchiveStatistics.QD_SCORE: ", eV/atom",
        }
        return unit_dict[metric]
