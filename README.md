[![Codacy Badge](https://api.codacy.com/project/badge/Grade/786b7822138a462c9e34f3cddcc89be6)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=JohnGiorgi/deidentified-cohort-identification-neuroips-workshop&amp;utm_campaign=Badge_Grade)

# DrBERT

A BERT-based model for end-to-end neural joint deidentification and cohort building from medical notes.

## Usage

### Train the model

```python
python -m drbert.run \
--dataset_folder ./data \
--model_name_or_path bert-base-cased \
--output_dir ./output \
--max_seq_length 364 \
--do_train \
--evaluate_during_training \
--train_batch_size 16 \
--eval_batch_size 256 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--warmup 0.1 \
--num_train_epochs 5 \
--logging_steps 3000 \
--save_steps 3000 \
--overwrite_output_dir \
--fp16
```