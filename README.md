[![Codacy Badge](https://api.codacy.com/project/badge/Grade/786b7822138a462c9e34f3cddcc89be6)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=JohnGiorgi/deidentified-cohort-identification-neuroips-workshop&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/786b7822138a462c9e34f3cddcc89be6)](https://www.codacy.com?utm_source=github.com&utm_medium=referral&utm_content=JohnGiorgi/deidentified-cohort-identification-neuroips-workshop&utm_campaign=Badge_Coverage)

# Dr. BERT

A transformer-based model for extensive multi-tasking of various clinical NLP tasks.

## Installation

### With pip

To install straight from GitHub using pip:

```
pip install -e https://github.com/JohnGiorgi/drbert.git
```

### From source (with development dependencies)

To install from source with development dependencies:

```
git clone https://github.com/JohnGiorgi/drbert.git
cd drbert
pip install -e .[dev]
```

### Other installation requirements

Regardless of the installation method, you will need to additionally download a [SpaCy](https://spacy.io/usage) language model:

```
$ python -m spacy download en_core_web_md
```

For GPU support, make sure to install PyTorch 1.2.0+ with CUDA support. See [here](https://pytorch.org/get-started/locally/) for instructions.

Finally, for mixed-precision training, you will need to install Apex. See [here](https://github.com/NVIDIA/apex) for instructions.

## Usage

```
python -m drbert.run \
--dataset_folder ./path/to/dataset \
--model_name_or_path bert-base-cased \
--output_dir ./output \
--max_seq_length 364 \
--do_train \
--evaluate_during_training \
--train_batch_size 16 \
--eval_batch_size 256 \
--learning_rate 2e-5 \
--weight_decay 0.1 \
--warmup 0.1 \
--num_train_epochs 5 \
--logging_steps 3000 \
--save_steps 3000 \
--overwrite_output_dir \
--fp16
```

Call

```
python -m drbert.run --help
```

for more information on the possible arguments.

## Testing

A test suite is included in the [tests folder](https://github.com/JohnGiorgi/drbert/tree/master/drbert/tests). To run the tests, [install from source with development dependencies](#from-source-with-development-dependencies).

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
