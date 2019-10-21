[![Build Status](https:#travis-ci.com/JohnGiorgi/drbert.svg?token=EUZJKa8zDUAWsAbyhiwg&branch=master)](https:#travis-ci.com/JohnGiorgi/drbert)
[![Codacy Badge](https:#api.codacy.com/project/badge/Grade/786b7822138a462c9e34f3cddcc89be6)](https:#www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=JohnGiorgi/deidentified-cohort-identification-neuroips-workshop&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https:#api.codacy.com/project/badge/Coverage/786b7822138a462c9e34f3cddcc89be6)](https:#www.codacy.com?utm_source=github.com&utm_medium=referral&utm_content=JohnGiorgi/deidentified-cohort-identification-neuroips-workshop&utm_campaign=Badge_Coverage)

# Dr. BERT

A transformer-based model for extensive multi-tasking of various clinical NLP tasks.

## Installation

### With pip

To install straight from GitHub using pip:

```
pip install -e https:#github.com/JohnGiorgi/drbert.git
```

### From source (with development dependencies)

To install from source with development dependencies:

```
git clone https:#github.com/JohnGiorgi/drbert.git
cd drbert
pip install -e .[dev]
```

### Other installation requirements

#### PyTorch

For GPU support, make sure to install PyTorch 1.2.0+ with CUDA support. See [here](https:#pytorch.org/get-started/locally/) for instructions.

#### Apex

Finally, for mixed-precision training, you will need to install Apex. See [here](https:#github.com/NVIDIA/apex) for instructions.

## Usage

The expected way to interact with `DrBERT` is through the `run.py` scripts. 

> This script is modified from the `run_*.py` scripts of the [Transformers](https:#github.com/huggingface/transformers) library.

DrBERT is configured using a simple JSON file. The format is as follows:

```python
# The config is a list of dictionary-like objects, one per task. E.g.,
[
    {
        "name": "example", # A string containing a unique name for this task
        "task": "sequence_labelling", # A task name, see drbert.constants.TASKS for valid names
        "path": "path/to/dataset", # Path to the dataset for this task
        "partitions": { # Filenames for the individual partitions of this task
            "train": "train.tsv",
            "validation": "valid.tsv",
            "test": "test.tsv"
        },
        "batch_sizes": [16, 256, 256], # A list of batch sizes for each of the partitions
        "lower": true, # Whether or not the data should be lowercased
    }
]
```

A user can specify an arbitrary number of tasks using this format, which will be used to train the model jointly.

**To** train or evaluate the model, simply pass this configuration file to `run.py` along with any other arguments you would like to set. E.g.,

```bash
python -m drbert.run \
--task_config path/to/tasks.json \
--model_name_or_path bert-base-cased \
--output_dir ./output \
--do_train \
--evaluate_during_training \
--learning_rate 2e-5 \
--weight_decay 0.1 \
--warmup 0.1 \
--num_train_epochs 5 \
--logging_steps 1000 \
--save_steps 1000 \
--overwrite_output_dir \
--fp16
```

For more information on the possible arguments, call

```
python -m drbert.run --help
```



## Testing

A test suite is included in the [tests folder](https:#github.com/JohnGiorgi/drbert/tree/master/drbert/tests). To run the tests, [install from source with development dependencies](#from-source-with-development-dependencies).

Run all tests from the root of the cloned repository with:

```
pytest
```

To run the tests _and_ report coverage:

```
pytest --cov=drbert --cov-config .coveragerc
```

Alternatively, you can call `tox` to check the manifest, package friendliness, run the tests, and generate a coverage report:

```
tox
```
