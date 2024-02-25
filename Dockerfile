FROM ubuntu:20.04 as base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    SETUPTOOLS_USE_DISTUTILS=stdlib

ENV POETRY_NO_INTERACTION=1

RUN apt update && apt upgrade --no-install-recommends -y

# install curl to allow pip and poetry installation
RUN apt-get install -y --no-install-recommends curl

# install python3.9
RUN apt-get install --no-install-recommends -y python3.9 python3.9-dev python3-distutils python3.9-venv
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 0

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

FROM base as code

WORKDIR /workdir

RUN apt-get install -y python3-vtk7

# copy required files into docker image
COPY pyproject.toml /workdir/pyproject.toml
COPY poetry.lock /workdir/poetry.lock

RUN poetry install --no-root --without dev

COPY README.md /workdir/README.md
COPY qd4csp /workdir/qd4csp
COPY mp_reference_analysis /workdir/mp_reference_analysis

RUN poetry install --without dev
