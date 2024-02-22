import shutil
from zipfile import ZipFile

import gdown
import requests

from qd4csp.utils.env_variables import EXPERIMENT_FOLDER


def download_data_folder():
    url_reported_results = "https://drive.google.com/file/d/1JPjPoRUQYWzkjBewPQIVlZYtMcG3eVyp/view?usp=share_link"
    filename = "reported_results.zip"
    output_path = str(EXPERIMENT_FOLDER / filename)
    gdown.download(url_reported_results, output_path, quiet=False, fuzzy=True)

    with ZipFile(output_path, 'r') as zObject:
        zObject.extractall(path=EXPERIMENT_FOLDER)

    shutil.rmtree(EXPERIMENT_FOLDER / "__MACOSX")
    shutil.rmtree(EXPERIMENT_FOLDER / filename)
