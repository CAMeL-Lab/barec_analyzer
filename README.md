# BAREC Analyzer

This repository contains scripts for preprocessing, training and evaluating the models in our paper [A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](https://arxiv.org/abs/2502.13520).
The BAREC corpus is avaialble on [huggingface](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0).

## Repository Structure
- `scripts/preprocess.py`: Processes raw texts into our tokenized input vriants (`Word`, `D3Tok`, `Lex`, and `D3Lex`). You **DO NOT** need to run this script to process BAREC corpus as we already provided these input variants for the full corpus.
- `scripts/train.py`: Script for fine-tuning pre-trained uning BAREC corpus. The script supports different loss functions and input variants. It also generates results and saves trained models.
- `scripts/collect_results.py`: Aggregates evaluation results from multiple trained models and exports them as CSV files for further analysis.

## Setup

**Install dependencies:**
To run `scripts/preprocess.py` you need to install [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) and get the CAMeLBERT MSA morphosyntactic tagger from `camel_data`.

```sh
git clone https://github.com/CAMeL-Lab/camel_tools.git
cd camel_tools

conda create -n camel_tools python=3.9
conda activate camel_tools

pip install -e .
camel_data -i disambig-bert-unfactored-msa
```

To run `scripts/train.py` and `scripts/collect_results.py`:

```sh
git clone https://github.com/CAMeL-Lab/barec_analyzer.git
cd barec_analyzer

conda create -n barec python=3.9
conda activate barec

pip install -r requirements.txt
```


## Usage

### Pre-processing

Preprocess raw text to different input variants.
You **DO NOT** need this script if you want to train on BAREC corpus as we already provided these input variants for the full corpus.

```sh
python scripts/preprocess.py \
  --input <INPUT_TXT_PATH> \
  --input_var <INPUT_VARIANT> \
  --db <MORPHOLOGY_DATABASE> \
  --output <OUTPUT_TXT_PATH>
```

- `--input`: Path to input text file containing raw text data
- `--input_var`: Input variant (`Word`, `D3Tok`, `Lex`, or `D3Lex`)
- `--db` (**optional**): Path to morphological database to use for processing
- `--output`: Path to output file to save processed text data

* **Important Note**: The default morphological analyzer used in the pre-processing script is not the same as the one in the paper, which is licensed by LDC. To download the same morphogical analyzer, you need to:

1. Optain the morphological analyzer from LDC ([LDC2010L01](https://catalog.ldc.upenn.edu/LDC2010L01)).
2. Download the muddled version of the analyzer from [here](https://github.com/CAMeL-Lab/CAMeLBERT_morphosyntactic_tagger/releases/download/v0.0.1/analyzer-msa.muddle).
3. Install [Muddler](https://github.com/CAMeL-Lab/muddler), a tool for sharing derived data, and use it to unmuddle the encrypted file.
  ```sh
  pip install muddler
  muddler unmuddle -s /PATH/TO/LDC2010L01.tgz -m /PATH/TO/analyzer-msa.muddle /PATH/TO/almor-s31.db.utf8
  ```

4. To use this analyzer in `scripts/preprocess.py`, pass it as a parameter (`--db "/PATH/TO/almor-s31.db.utf8"`).


### Training a Model

Run the training script on BAREC corpus with configurable parameters:

```sh
python scripts/train.py \
  --loss <LOSS_TYPE> \
  --model <MODEL_CHECKPOINT> \
  --input_var <INPUT_TYPE> \
  --save_dir <MODEL_SAVE_BASE_DIR> \
  --output_path <OUTPUT_XLSX_DIR>
```

- `--loss`: Loss function (e.g., `CE`, `EMD`, `OLL1`, etc.)
- `--model`: Model checkpoint (e.g., HuggingFace model name or path)
- `--input_var`: Input variant (`Word`, `D3Tok`, `Lex`, or `D3Lex`)
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
