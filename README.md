

## Install
Clone the repository.

Then download the data here:
https://polybox.ethz.ch/index.php/s/oFZhVdO9Jdcq3lm
and copy the contents into the subdirectory ./data/

Then create and environment and install the dependencies

    conda create --name dvs_tracker cython
    conda activate dvs_tracker
    pip install -r requirements.txt

## Run
    python dvs_tracker/main_brian2.py

