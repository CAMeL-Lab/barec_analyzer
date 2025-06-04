# BAREC Corpus

This repository contains scripts for training and evaluating models on the BAREC corpus, a large-scale dataset for Arabic Readability Assessment.

## Repository Structure

- `scripts/train_OLL.py`: Main training script supporting various loss functions and flexible model configuration. Allows training with different checkpoints, loss types, and input text types. Outputs results and saves trained models.
- `scripts/collect_results.py`: Aggregates evaluation results from multiple trained models and exports them as CSV files for further analysis.

## Setup

**Install dependencies:**
   ```sh
   git clone https://github.com/CAMeL-Lab/barec_corpus.git
   cd barec_corpus

   conda create --name barec --file requirements.txt
   ```


## Usage

### Training a Model

Run the training script with configurable parameters:

```sh
python scripts/train_OLL.py \
  --loss <LOSS_TYPE> \
  --checkpoint <MODEL_CHECKPOINT> \
  --input <INPUT_TYPE> \
  --save_dir <MODEL_SAVE_BASE_DIR> \
  --output_path <OUTPUT_XLSX_DIR>
```

- `--loss`: Loss type (e.g., `CE`, `EMD`, `OLL1`, etc.)
- `--checkpoint`: Model checkpoint (e.g., HuggingFace model name or path)
- `--input`: Input text type (e.g., `word_sents`)
- `--save_dir`: Base directory for saving trained model folders
- `--output_path`: Directory to save output XLSX files

### Collecting Results

After training multiple models, aggregate their results:

```sh
python scripts/collect_results.py \
  --models_path <MODELS_DIR> \
  --output_path <RESULTS_CSV_DIR>
```

- `--models_path`: Directory containing all trained model folders
- `--output_path`: Directory to save the aggregated CSV files

## License
See the `LICENSE` file for license information.