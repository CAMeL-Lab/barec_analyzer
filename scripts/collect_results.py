import os
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Collect results from model directories and save as CSV.")
parser.add_argument('--models_path', type=str, required=True, help='Path to the directory containing all models folders.')
parser.add_argument('--output_path', type=str, required=True, help='Directory to save the resulting CSV files.')
args = parser.parse_args()

models_path = args.models_path
output_path = args.output_path

trained_models = os.listdir(models_path)
trained_models.sort()

dev_results = {
    "model": [],
    "accuracy": [],
    "adj_accuracy": [],
    "distance": [],
    "QWK": [],
    "accuracy_7": [],
    "accuracy_5": [],
    "accuracy_3": []
}

test_results = {
    "model": [],
    "accuracy": [],
    "adj_accuracy": [],
    "distance": [],
    "QWK": [],
    "accuracy_7": [],
    "accuracy_5": [],
    "accuracy_3": []
}

ignore_models = [
    "bert-base-arabertv2_d3tok_sents_regression_default_19levels",
    "bert-base-arabertv2_d3tok_sents_CE_default_3levels",
    "bert-base-arabertv2_d3tok_sents_CE_default_5levels",
    "bert-base-arabertv2_d3tok_sents_CE_default_7levels",
    ".DS_Store"
]

for model in trained_models:
    if model in ignore_models:
        continue
    results_path = os.path.join(models_path, model, "all_results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    
    dev_results["model"].append(model)
    dev_results["accuracy"].append(results["eval_accuracy"])
    dev_results["adj_accuracy"].append(results["eval_accuracy_with_margin"])
    dev_results["accuracy_7"].append(results["eval_accuracy_7"])
    dev_results["accuracy_5"].append(results["eval_accuracy_5"])
    dev_results["accuracy_3"].append(results["eval_accuracy_3"])
    dev_results["distance"].append(results["eval_Distance"])
    dev_results["QWK"].append(results["eval_Quadratic Weighted Kappa"])
    
    test_results["model"].append(model)
    test_results["accuracy"].append(results["test_accuracy"])
    test_results["adj_accuracy"].append(results["test_accuracy_with_margin"])
    test_results["accuracy_7"].append(results["test_accuracy_7"])
    test_results["accuracy_5"].append(results["test_accuracy_5"])
    test_results["accuracy_3"].append(results["test_accuracy_3"])
    test_results["distance"].append(results["test_Distance"])
    test_results["QWK"].append(results["test_Quadratic Weighted Kappa"])

dev_df = pd.DataFrame.from_dict(dev_results)
test_df = pd.DataFrame.from_dict(test_results)
dev_df.to_csv(os.path.join(output_path, "results_dev.csv"), index=False)
test_df.to_csv(os.path.join(output_path, "results_test.csv"), index=False)