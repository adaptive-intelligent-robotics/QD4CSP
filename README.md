# QD4CSP: Quality Diversity for Crystal Structure Prediction
<p align="center">
<img src="images/cvt_plot_gif.gif" height="400" width="450">
</p>


`QD4CSP` is the first of its kind implementation combining the strengths of Quality-Diversity algorithms
for inorganic crystal structure prediction. 
This repository contains the code used for the paper _Illuminating the property space in crystal structure prediction using 
Quality-Diversity algorithms_.

The gif shows the evolution of an archive of $TiO_2$ crystal structures over 5000 evaluations.

## Getting started with the package
### Installation
To get started with this package clone this repo:

```bash
git clone https://github.com/adaptive-intelligent-robotics/QD4CSP
```
Then enter the correct directory on your machine:
```bash
cd QD4CSP
```
We provide two installation methods, one using poetry (preferred) and using standard `requirements.txt`.

#### Option 1: Poetry
This package uses [poetry](https://python-poetry.org) dependency manager. 
To install all dependencies run:
```bash
poetry install
```

#### Option 2: Python virtual environments
Once you have cloned the repository, create and activate your virtual environment:
```shell
python3 -m venv ./venv
source venv/bin/activate
```
Then install the requirements:
```shell script
pip3 install -r requirements.txt
```

### External integrations set up
#### [Required] Materials Project
For comparison to reference structures we use the Materials Project API.
Prior to running the code you must [set up your own account](https://next-gen.materialsproject.org) on the Materials Project and 
get the api key from your dashboard following the instructions [here](https://next-gen.materialsproject.org/api).

Then add your API key as an environment variable like so:
```shell script
export MP_API_KEY=<your-api-key>
```

#### [Optional] `CHGNet`
This step is optional but ensures significant speed improvements.

As guidelines were not available on the package at the time of writing we provide our method to ensure
cython is set up correctly to be used with `CHGNet`.

First clone the `CHGNet` repository and enter the folder
```shell
git clone https://github.com/CederGroupHub/chgnet.git
cd chgnet
```
Then run:
```shell
python3 setup.py build_ext --inplace
```
Now we will need to copy the generated filed into our virtual environment 

```shell
cd chgnet/graph
copy *.c venv/lib/chgnet/graph
copy *.pyx venv/lib/chgnet/graph
```

You can verify this by running a script containing the following:
```python
from chgnet.model import CHGNet

if __name__ == '__main__':
    model = CHGNet.load()
    print(model.graph_converter.algorithm)
```

### Using the package
To run a demo experiment run:
```shell
 python3 scripts/experiment_from_config.py experiment_configs/demo.json
```
This will run a very simple demo with 2 $TiO_2$ structures. All results will be saved under the `experiments` folder.

Experiments are most conveniently defined using a configuration file. 
These files can be generated individually or in batches using the directions below 

The configuration files used for the paper are also provided. 
To reproduce the experiments reported run the same command with any of the provided configuration files:
`TiO2_benchamrk.json`, `SiO2_like_benchmark.json`, `SiC_like_benchmark.json` and `C_like_benchmark.json`

```shell
 python3 scripts/experiments/experiment_from_config.py experiment_configs/<desired-config-file>
```

## Running an Experiment 
You can run an experiment from a configuration file or directly from a file. 
The latter is recommended for debugging new features. 

### Running from a configuration file
_If you are NOT using the $C$, $SiO_2$, $SiC$ or $TiO_2$ system refer to section New materials set up below._

To run your job simply run

```shell
python3 scripts/experiment_from_config.py experiment_configs/<your-config-name>.json
```

Or if you prefer to change parameters directly in a python script you can amend them in `csp_scripts/`
### Running Feature Debugging Script
```shell
python3 scripts/run_experiment.py  
```

### Generating Configuration Files
To generate a configuration file for your experiment simply run
```shell
python3 scripts/generate_pre_filled_config.py <config_filename> <config_folder>
```
Passing the `<config_folder>` parameter will create a subfolder within `experiment_configs`.
If it is not passed the config file will save directly within `experiment_configs`.

This will be filled with some default values and the resulting json should then be updated directly.

### New materials set up 
To use the repo on a new material first you should set up the reference data first. 

To do this simply update the scripts/prepare_reference_data.py file with your desired system. 
Then run:
```shell
python3 scripts/prepare_reference_data.py
```

### [Optional] Setting up environment variables
This repo relies on 4 environment variables:
* `EXPERIMENT_FOLDER`
* `MP_REFERENCE_FOLDER`
* `CONFIGS_FOLDER`
* `MP_API_KEY`

The former 3 are set up with defaults to save the experiments in the following structure:
```shell
├── experiment_configs #CONFIGS_FOLDER
│  ├── C_like_benchmark.json
│  ├── demo.json
│  ├── SiC_like_benchmark.json
│  ├── SiO2_like_benchmark.json
│  ├── TiO2_benchmark.json
├── experiments # EXPERIMENT_FOLDER
│  ├── centroids
│  │  ├── centroids_200_2_C_band_gap_0_1_shear_modulus_0_1.dat
│  │  ├── centroids_200_2_SiO2_band_gap_0_1_shear_modulus_0_1.dat
│  │  ├── centroids_200_2_band_gap_0_1_shear_modulus_0_1.dat
├── mp_reference_analysis # MP_REFERENCE_FOLDER
│  ├── C_24
│  ├── SiO2_24
│  └── TiC_24
│  └── TiO2_24
```
If desired please set the necessary environment variables using 
```shell
export <env-variable-name>=<directory-location>
```
or your preferred method of setting environment variables. 
