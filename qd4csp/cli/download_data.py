#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import shutil
from pathlib import Path
from zipfile import ZipFile

import gdown
import requests

from qd4csp.utils.env_variables import EXPERIMENT_FOLDER


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_data_folder():
    url_reported_results = "https://drive.google.com/file/d/1gi6a2me43S09jvjvP2TE_Cq-5RqyvwgL/view?usp=share_link" #todo: update
    output_path = EXPERIMENT_FOLDER / "reported_results.zip"
    gdown.download(url_reported_results, output_path, quiet=False, fuzzy=True)

    with ZipFile(output_path, 'r') as zObject:
        zObject.extractall(path=EXPERIMENT_FOLDER)

    shutil.rmtree(EXPERIMENT_FOLDER / "__MACOSX")
