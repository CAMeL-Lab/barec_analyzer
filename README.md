# BAREC Analyzer

This repository contains scripts for training and evaluating the models in our paper [A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](https://arxiv.org/abs/2502.13520).

## Repository Structure

- `scripts/train.py`: Main training script supporting different loss functions and input variants. It also generates results and saves trained models.
- `scripts/collect_results.py`: Aggregates evaluation results from multiple trained models and exports them as CSV files for further analysis.

## Setup

**Install dependencies:**
   ```sh
   git clone https://github.com/CAMeL-Lab/barec_corpus.git
   cd barec_corpus

   conda create --name barec --file requirements.txt
   conda activate barec
   ```


## Usage

### Training a Model

Run the training script with configurable parameters:

```sh
python scripts/train.py \
  --loss <LOSS_TYPE> \
  --model <MODEL_CHECKPOINT> \
  --input_var <INPUT_TYPE> \
  --save_dir <MODEL_SAVE_BASE_DIR> \
  --output_path <OUTPUT_XLSX_DIR>
```

- `--loss`: Loss type (e.g., `CE`, `EMD`, `OLL1`, etc.)
- `--model`: Model checkpoint (e.g., HuggingFace model name or path)
- `--input_var`: Input text type (e.g., `Word`)
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

## Citation
```
@inproceedings{elmadani-etal-2025-readability,
    title = "A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment",
    author = "Elmadani, Khalid N.  and
      Habash, Nizar  and
      Taha-Thomure, Hanada",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics"
}
```

## License
See the `LICENSE` file for license information.
