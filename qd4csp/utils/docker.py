from pathlib import Path


def environment_is_docker():
    return Path('/.dockerenv').is_file()
