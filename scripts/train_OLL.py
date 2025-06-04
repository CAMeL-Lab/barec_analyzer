from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer, set_seed
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score , recall_score, mean_absolute_error, cohen_kappa_score
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import argparse
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train model with configurable parameters")
parser.add_argument('--loss', type=str, required=True, help='Loss type (e.g., CE, EMD)')
parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
parser.add_argument('--input', type=str, required=True, help='Input text type (e.g., word_sents)')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the trained model')
parser.add_argument('--output_path', type=str, required=True, help='Directory to save the output xlsx files')
args = parser.parse_args()

loss_type = args.loss
checkpoint = args.checkpoint
input_text = args.input
save_dir_base = args.save_dir
output_path = args.output_path

barec_7_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 4, 12: 5, 13: 5, 14: 6, 15: 6, 16: 7, 17: 7, 18: 7, 19: 7
}
barec_5_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 4, 15: 4, 16: 5, 17: 5, 18: 5, 19: 5
}
barec_3_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3
}

# Compose the full save_dir using the base and dynamic naming
save_dir = os.path.join(save_dir_base, f"{checkpoint.split('/')[-1]}_{input_text}_{loss_type}_default_19levels")

dev_out_xlsx = os.path.join(output_path, f"dev_{checkpoint.split('/')[-1]}_{input_text}_{loss_type}_default_19levels.xlsx")
test_out_xlsx = os.path.join(output_path, f"test_{checkpoint.split('/')[-1]}_{input_text}_{loss_type}_default_19levels.xlsx")

print(f"loss: {loss_type}, model: {checkpoint.split('/')[-1]}, input_text: {input_text}, d_mat_type: default, levels: 19")

d_mat_type = "default"
n_levels = 19

data_path = 'data/1M_sentences_v1_morph_clean.xlsx'

if d_mat_type == "default":
    d_matrix =  [[abs(i-j) for i in range(n_levels)] for j in range(n_levels)]
else:
    d_matrix = [[(abs(i-j)/18)+(abs(barec_7_dict[i+1]-barec_7_dict[j+1])/6)+(abs(barec_5_dict[i+1]-barec_5_dict[j+1])/4)+(abs(barec_3_dict[i+1]-barec_3_dict[j+1])/2) for i in range(19)] for j in range(19)]

class OLL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*abs(distances_tensor)**2
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class nOLL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[dist_matrix[true_labels[j][i]][label_ids[j][i]]/np.sum([dist_matrix[n][label_ids[j][i]] for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*abs(distances_tensor)**2
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class WKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        y_pred = F.softmax(logits,dim=1)
        label_vec = torch.range(0,num_classes-1, dtype=torch.float)
        row_label_vec = torch.tensor(torch.reshape(label_vec, (1, num_classes)), requires_grad=True,device='cuda:0')
        col_label_vec = torch.tensor(torch.reshape(label_vec, (num_classes, 1)), requires_grad=True,device='cuda:0')
        col_mat = torch.tile(col_label_vec, (1, num_classes))
        row_mat = torch.tile(row_label_vec, (num_classes, 1))
        weight_mat = (col_mat - row_mat) ** 2
        y_true = torch.tensor(F.one_hot(labels, num_classes=num_classes), dtype=col_label_vec.dtype, requires_grad=True)
        batch_size = y_true.shape[0]
        cat_labels = torch.matmul(y_true, col_label_vec)
        cat_label_mat = torch.tensor(torch.tile(cat_labels, [1, num_classes]), requires_grad=True,device='cuda:0')
        row_label_mat = torch.tensor(torch.tile(row_label_vec, [batch_size, 1]), requires_grad=True,device='cuda:0')
        
        weight = (cat_label_mat - row_label_mat) ** 2
        numerator = torch.sum(weight * y_pred)
        label_dist = torch.sum(y_true, axis=0)
        pred_dist = torch.sum(y_pred, axis=0)
        w_pred_dist = torch.t(torch.matmul(weight_mat, pred_dist))
        denominator = torch.sum(torch.matmul(label_dist, w_pred_dist/batch_size),axis = 0)
        loss = torch.log(numerator/denominator + 1e-7)
 
        return (loss, outputs) if return_outputs else loss

class SOFT10Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-10*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-10*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT5Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-5*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-5*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-2*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-2*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT3Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-3*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-3*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT4Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-4*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-4*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT5Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-5*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-5*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT10Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-10*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-10*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT30Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-30*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-30*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class OLL1Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class OLL15Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(1.5)
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class OLL05Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(0.5)
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class OLL025Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(0.25)
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss
        
class OLL075Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(0.75)
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss
        
class OLL01Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(0.1)
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss
    
    
class EMDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.num_labels
        dist_matrix = d_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        CDF_pred = torch.cumsum(probas,dim=1)
        CDF_true = torch.tensor([labels[k].item()*[0.] + (num_classes-labels[k].item())*[1.] for k in range(len(labels))],device='cuda:0', requires_grad=True)
        err = (CDF_pred - CDF_true)**2
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss


losses_dict = {"CE": Trainer,
               "OLL1": OLL1Trainer,
               "OLL15": OLL15Trainer,
               "OLL2": OLL2Trainer,
               "OLL05": OLL05Trainer,
               "OLL025": OLL025Trainer,
               "OLL075": OLL075Trainer,
               "OLL01": OLL01Trainer,
               "nOLL2": nOLL2Trainer,
               "WKL": WKLTrainer,
               "SOFT2": SOFT2Trainer,
               "SOFT3": SOFT3Trainer,
               "SOFT4": SOFT4Trainer,
               "SOFT5": SOFT5Trainer,
               "SOFT10": SOFT10Trainer,
               "SOFT30": SOFT30Trainer,
               "EMD": EMDTrainer,
               "CORAL": Trainer}

loss_function = losses_dict[loss_type]


all_df = pd.read_excel(data_path, header=0)

DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

real_names = {
    input_text: DATA_COLUMN,
    'RL_num_'+str(n_levels): LABEL_COLUMN
}

all_df.rename(columns= real_names, inplace=True)


minus_mapper = {}
for i in range(n_levels):
  minus_mapper[i+1] = i
all_df = all_df.replace({LABEL_COLUMN: minus_mapper})


all_df = all_df.groupby('Split')

all_df = all_df[[DATA_COLUMN, LABEL_COLUMN]]
all_df.columns = [DATA_COLUMN, LABEL_COLUMN]

train_df = all_df.get_group('Train')
dev_df = all_df.get_group('Dev')
test_df = all_df.get_group('Test')

set_seed(42)


train = datasets.Dataset.from_pandas(train_df)
dev = datasets.Dataset.from_pandas(dev_df)
test = datasets.Dataset.from_pandas(test_df)
dataset = load_dataset("labr") #dump loading .. only to match the dataset template from huggingface
dataset['train'] = train
dataset['dev'] = dev
dataset['test'] = test
dataset


tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example[DATA_COLUMN], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets


def model_init():
  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = n_levels, ignore_mismatched_sizes=True)
  model.dist_matix = [[abs(i-j) for i in range(model.num_labels)] for j in range(model.num_labels)]
  for param in model.parameters(): param.data = param.data.contiguous()
  return model


def accuracy_with_margin_classification(y_true, y_pred, margin=1):
    """
    Calculates accuracy with a specified margin of error for classification.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        margin (int): Acceptable difference between true and predicted classes.

    Returns:
        float: Accuracy with margin.
    """
    correct_predictions = np.sum(np.abs(np.array(y_true) - np.array(y_pred)) <= margin)
    return correct_predictions / len(y_true)

def compute_metrics(p):
  preds = np.argmax(p.predictions, axis=1)
  assert len(preds) == len(p.label_ids)
  print(classification_report(p.label_ids,preds,digits=4))
  print(confusion_matrix(p.label_ids,preds))
  preds_19 = list(preds)
  labels_19 = list(p.label_ids)
  preds_7 = [barec_7_dict[i+1] for i in preds_19]
  labels_7 = [barec_7_dict[i+1] for i in labels_19]
  preds_5 = [barec_5_dict[i+1] for i in preds_19]
  labels_5 = [barec_5_dict[i+1] for i in labels_19]
  preds_3 = [barec_3_dict[i+1] for i in preds_19]
  labels_3 = [barec_3_dict[i+1] for i in labels_19]

  macro_f1 = f1_score(p.label_ids,preds,average='macro')
  macro_precision = precision_score(p.label_ids,preds,average='macro')
  macro_recall = recall_score(p.label_ids,preds,average='macro')
  acc = accuracy_score(p.label_ids,preds)
  acc_with_margin = accuracy_with_margin_classification(p.label_ids, preds, margin=1)
  acc_7 = accuracy_score(labels_7,preds_7)
  acc_5 = accuracy_score(labels_5,preds_5)
  acc_3 = accuracy_score(labels_3,preds_3)
  QWK = cohen_kappa_score(p.label_ids, preds, weights='quadratic')
  dist = mean_absolute_error(p.label_ids, preds)
  return {
      'macro_f1' : macro_f1,
      'macro_precision': macro_precision,
      'macro_recall': macro_recall,
      'accuracy': acc,
      'accuracy_with_margin': acc_with_margin,
      'Distance': dist,
      'Quadratic Weighted Kappa': QWK,
      'accuracy_7': acc_7,
      'accuracy_5': acc_5,
      'accuracy_3': acc_3
  }


training_args = TrainingArguments(save_dir,
                                  evaluation_strategy="epoch",
                                  num_train_epochs=6,
                                  per_device_train_batch_size= 64,
                                  per_device_eval_batch_size=16,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="eval_loss",
                                  greater_is_better=False,
                                  save_strategy="epoch",
                                  #overwrite_output_dir=True,
                                  #save_steps=496,
                                  save_total_limit=1,
                                  #learning_rate=lr
                                  )



trainer = loss_function(model_init=model_init,
                  args = training_args,
                  train_dataset = tokenized_datasets['train'],
                  eval_dataset = tokenized_datasets['dev'],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics = compute_metrics)


trainer.train()


preds, labels, metrics = trainer.predict(tokenized_datasets['dev'], metric_key_prefix="eval")
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)



plus_mapper = {}
for i in range(19):
  plus_mapper[i] = i+1

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rank_simple(vector):
    return [i+1 for i in sorted(range(len(vector)), key=vector.__getitem__, reverse=True)]


probs = {}
for i in range(19):
  probs[i+1] = []
ranks = {}
for i in range(19):
  ranks[i+1] = []
texts = []
original_texts = []
labels = []
predictions = []
labels_7 = []
predictions_7 = []
labels_5 = []
predictions_5 = []
labels_3 = []
predictions_3 = []
is_equal = []
is_equal_7 = []
is_equal_5 = []
is_equal_3 = []
within_one = []
within_top_2 = []
within_top_3 = []
within_top_4 = []
rank_of_correct = []
diff = []

for i in range(len(preds)):
  texts.append(list(dev_df['text'])[i])
  labels.append(list(dev_df['label'])[i]+1)
  predictions.append(argmax(preds[i])+1)
  labels_7.append(barec_7_dict[labels[-1]])
  predictions_7.append(barec_7_dict[predictions[-1]])
  labels_5.append(barec_5_dict[labels[-1]])
  predictions_5.append(barec_5_dict[predictions[-1]])
  labels_3.append(barec_3_dict[labels[-1]])
  predictions_3.append(barec_3_dict[predictions[-1]])
  softs = softmax(preds[i])
  rank = rank_simple(softs)

  if labels[-1] == predictions[-1]:
    is_equal.append(1)
  else:
    is_equal.append(0)
    
  if labels_7[-1] == predictions_7[-1]:
    is_equal_7.append(1)
  else:
    is_equal_7.append(0)
    
  if labels_5[-1] == predictions_5[-1]:
    is_equal_5.append(1)
  else:
    is_equal_5.append(0)
    
  if labels_3[-1] == predictions_3[-1]:
    is_equal_3.append(1)
  else:
    is_equal_3.append(0)

  if abs(labels[-1] - predictions[-1]) <= 1:
    within_one.append(1)
  else:
    within_one.append(0)

  if labels[-1] in rank[:2]:
    within_top_2.append(1)
  else:
    within_top_2.append(0)

  if labels[-1] in rank[:3]:
    within_top_3.append(1)
  else:
    within_top_3.append(0)

  if labels[-1] in rank[:4]:
    within_top_4.append(1)
  else:
    within_top_4.append(0)

  rank_of_correct.append(rank.index(labels[-1])+1)
  diff.append(max([labels[-1],predictions[-1]])-min([labels[-1],predictions[-1]]))
  for j in range(19):
    probs[j+1].append(softs[j])
    ranks[j+1].append(rank[j])


QWK = cohen_kappa_score(labels, predictions, weights='quadratic')
acc = sum(is_equal)/len(is_equal)
acc_7 = sum(is_equal_7)/len(is_equal_7)
acc_5 = sum(is_equal_5)/len(is_equal_5)
acc_3 = sum(is_equal_3)/len(is_equal_3)
acc_within_one_level = sum(within_one)/len(within_one)
acc_top_2 = sum(within_top_2)/len(within_top_2)
acc_top_3 = sum(within_top_3)/len(within_top_3)
acc_top_4 = sum(within_top_4)/len(within_top_4)
avg_rank = sum(rank_of_correct)/len(rank_of_correct)
avg_distance = sum(diff)/len(diff)


print(f"Accuracy: {acc*100:.4f}")
print(f"Accuracy with margin of one level: {acc_within_one_level*100:.4f}")
#print(f"Accuracy of top 2 choices: {acc_top_2*100:.4f}")
#print(f"Accuracy of top 3 choices: {acc_top_3*100:.4f}")
#print(f"Accuracy of top 4 choices: {acc_top_4*100:.4f}")
#print(f"Accuracy of SAMER levels: {acc_samer*100:.4f}")
#print(f"Accuracy of BAREC groups: {acc_barec*100:.4f}")
#print(f"Average rank of correct label: {avg_rank:.4f}")
print(f"Average distance between labels and predictions: {avg_distance:.6f}")
print(f"Quadratic Weighted Kappa: {QWK*100:.4f}")
print(f"Accuracy_7: {acc_7*100:.4f}")
print(f"Accuracy_5: {acc_5*100:.4f}")
print(f"Accuracy_3: {acc_3*100:.4f}")


v = {
    #'original text': original_texts,
    'text': texts,
    'label': labels,
    'prediction': predictions,
    'is_equal': is_equal,
    'within_one_level': within_one,
    'within_top2_ranks': within_top_2,
    'within_top3_ranks': within_top_3,
    'within_top4_ranks': within_top_4,
    'rank_of_correct_label': rank_of_correct,
    'diff': diff
}

for i in range(1,20):
  s = 'p'+str(i)
  v[s] = probs[i]
for i in range(1,20):
  s = 'rank'+str(i)
  v[s] = ranks[i]

final_df = pd.DataFrame.from_dict(v)
final_df.to_excel(dev_out_xlsx ,index=False)


############Test#################

preds, labels, metrics = trainer.predict(tokenized_datasets['test'], metric_key_prefix="test")
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)

plus_mapper = {}
for i in range(19):
  plus_mapper[i] = i+1

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rank_simple(vector):
    return [i+1 for i in sorted(range(len(vector)), key=vector.__getitem__, reverse=True)]



probs = {}
for i in range(19):
  probs[i+1] = []
ranks = {}
for i in range(19):
  ranks[i+1] = []
texts = []
original_texts = []
labels = []
predictions = []
labels_7 = []
predictions_7 = []
labels_5 = []
predictions_5 = []
labels_3 = []
predictions_3 = []
is_equal = []
is_equal_7 = []
is_equal_5 = []
is_equal_3 = []
within_one = []
within_top_2 = []
within_top_3 = []
within_top_4 = []
rank_of_correct = []
diff = []

for i in range(len(preds)):
  texts.append(list(test_df['text'])[i])
  labels.append(list(test_df['label'])[i]+1)
  predictions.append(argmax(preds[i])+1)
  labels_7.append(barec_7_dict[labels[-1]])
  predictions_7.append(barec_7_dict[predictions[-1]])
  labels_5.append(barec_5_dict[labels[-1]])
  predictions_5.append(barec_5_dict[predictions[-1]])
  labels_3.append(barec_3_dict[labels[-1]])
  predictions_3.append(barec_3_dict[predictions[-1]])
  softs = softmax(preds[i])
  rank = rank_simple(softs)

  if labels[-1] == predictions[-1]:
    is_equal.append(1)
  else:
    is_equal.append(0)
    
  if labels_7[-1] == predictions_7[-1]:
    is_equal_7.append(1)
  else:
    is_equal_7.append(0)
    
  if labels_5[-1] == predictions_5[-1]:
    is_equal_5.append(1)
  else:
    is_equal_5.append(0)
    
  if labels_3[-1] == predictions_3[-1]:
    is_equal_3.append(1)
  else:
    is_equal_3.append(0)

  if abs(labels[-1] - predictions[-1]) <= 1:
    within_one.append(1)
  else:
    within_one.append(0)

  if labels[-1] in rank[:2]:
    within_top_2.append(1)
  else:
    within_top_2.append(0)

  if labels[-1] in rank[:3]:
    within_top_3.append(1)
  else:
    within_top_3.append(0)

  if labels[-1] in rank[:4]:
    within_top_4.append(1)
  else:
    within_top_4.append(0)

  rank_of_correct.append(rank.index(labels[-1])+1)
  diff.append(max([labels[-1],predictions[-1]])-min([labels[-1],predictions[-1]]))
  for j in range(19):
    probs[j+1].append(softs[j])
    ranks[j+1].append(rank[j])


QWK = cohen_kappa_score(labels, predictions, weights='quadratic')
acc = sum(is_equal)/len(is_equal)
acc_7 = sum(is_equal_7)/len(is_equal_7)
acc_5 = sum(is_equal_5)/len(is_equal_5)
acc_3 = sum(is_equal_3)/len(is_equal_3)
acc_within_one_level = sum(within_one)/len(within_one)
acc_top_2 = sum(within_top_2)/len(within_top_2)
acc_top_3 = sum(within_top_3)/len(within_top_3)
acc_top_4 = sum(within_top_4)/len(within_top_4)
avg_rank = sum(rank_of_correct)/len(rank_of_correct)
avg_distance = sum(diff)/len(diff)


print(f"Accuracy: {acc*100:.4f}")
print(f"Accuracy with margin of one level: {acc_within_one_level*100:.4f}")
#print(f"Accuracy of top 2 choices: {acc_top_2*100:.4f}")
#print(f"Accuracy of top 3 choices: {acc_top_3*100:.4f}")
#print(f"Accuracy of top 4 choices: {acc_top_4*100:.4f}")
#print(f"Accuracy of SAMER levels: {acc_samer*100:.4f}")
#print(f"Accuracy of BAREC groups: {acc_barec*100:.4f}")
#print(f"Average rank of correct label: {avg_rank:.4f}")
print(f"Average distance between labels and predictions: {avg_distance:.6f}")
print(f"Quadratic Weighted Kappa: {QWK*100:.4f}")
print(f"Accuracy_7: {acc_7*100:.4f}")
print(f"Accuracy_5: {acc_5*100:.4f}")
print(f"Accuracy_3: {acc_3*100:.4f}")


v = {
    #'original text': original_texts,
    'text': texts,
    'label': labels,
    'prediction': predictions,
    'is_equal': is_equal,
    'within_one_level': within_one,
    'within_top2_ranks': within_top_2,
    'within_top3_ranks': within_top_3,
    'within_top4_ranks': within_top_4,
    'rank_of_correct_label': rank_of_correct,
    'diff': diff
}

for i in range(1,20):
  s = 'p'+str(i)
  v[s] = probs[i]
for i in range(1,20):
  s = 'rank'+str(i)
  v[s] = ranks[i]

final_df = pd.DataFrame.from_dict(v)
final_df.to_excel(test_out_xlsx ,index=False)




trainer.save_model(save_dir)
