import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import datasets 
import json
from dataloader import CoNLLDataset
from model import NERClassifier
from utils import save_checkpoint, log_gradient_norm
from tqdm import tqdm

from conlleval import evaluate
import itertools


def evaluate_model(model, dataloader, device, mode, step, class_mapping=None):
    """Evaluates the model performance."""
    if mode not in ["Train", "Validation"]:
        raise ValueError(
            f"Invalid value for mode! Expected 'Train' or 'Validation but received {mode}"
        )

    if class_mapping is None:
        raise ValueError("Argument @class_mapping not provided!")

    y_true_accumulator = []
    y_pred_accumulator = []

    print("Started model evaluation.")
    for x, y, padding_mask in dataloader:
        x, y = x.to(device), y.to(device)
        padding_mask = padding_mask.to(device)
        y_pred = model(x, padding_mask)

        # Extract predictions and labels only for pre-padding tokens
        unpadded_mask = torch.logical_not(padding_mask)
        y_pred = y_pred[unpadded_mask]
        y = y[unpadded_mask]

        y_pred = y_pred.argmax(dim=1)
        y_pred = y_pred.view(-1).detach().cpu().tolist()
        y = y.view(-1).detach().cpu().tolist()

        y_true_accumulator += y
        y_pred_accumulator += y_pred

    # Map the integer labels back to NER tags
    y_pred_accumulator = [class_mapping[str(pred)] for pred in y_pred_accumulator]
    y_true_accumulator = [class_mapping[str(pred)] for pred in y_true_accumulator]

    y_pred_accumulator = np.array(y_pred_accumulator)
    y_true_accumulator = np.array(y_true_accumulator)

    # Extract labels and predictions where target label isn't O
    non_O_ind = np.where(y_true_accumulator != "O")
    y_pred_non_0 = y_pred_accumulator[non_O_ind]
    y_true_non_0 = y_true_accumulator[non_O_ind]

    # Calculate and log accuracy
    accuracy_total = accuracy_score(y_true_accumulator, 
                                    y_pred_accumulator)
    accuracy_non_O = accuracy_score(y_true_non_0,
                                    y_pred_non_0)
    print(f"{mode}/Accuracy-Total",
                      accuracy_total, step)
    print(f"{mode}/Accuracy-Non-O",
                      accuracy_non_O, step)

    # Calculate and log F1 score
    f1_total = f1_score(y_true_accumulator,
                        y_pred_accumulator,
                        average="weighted")
    f1_non_O = f1_score(y_true_non_0,
                        y_pred_non_0,
                        average="weighted")
    print(f"{mode}/F1-Total",
                      f1_total, step)
    print(f"{mode}/F1-Non-O",
                      f1_non_O, step)

    print(classification_report(y_true_accumulator, y_pred_accumulator, digits=4))
    return f1_total

def final_evaluate_model(model, dataloader, device, mode, class_mapping=None):
    """Evaluates the model performance."""
    if mode not in ["Train", "Validation"]:
        raise ValueError(
            f"Invalid value for mode! Expected 'Train' or 'Validation but received {mode}"
        )

    if class_mapping is None:
        raise ValueError("Argument @class_mapping not provided!")

    y_true_list = []
    y_pred_list = []

    print("Started model evaluation.")
    for x, y, padding_mask in dataloader:
        x, y = x.to(device), y.to(device)
        padding_mask = padding_mask.to(device)
        y_pred = model(x, padding_mask)

        # Extract predictions and labels only for pre-padding tokens
        unpadded_mask = torch.logical_not(padding_mask)
        y_pred = y_pred[unpadded_mask]
        y = y[unpadded_mask]
        y_pred = y_pred.argmax(dim=1)
        
        for i in range(len(padding_mask)):
            predictions = []
            labels = []
            for j in range(len(padding_mask[i])):
                if padding_mask[j] == False:
                    predictions.append(class_mapping[y_pred[i][j]])
                    labels.append(class_mapping[y[i][j]])
                else:
                    break
            y_true_list.append(labels)
            y_pred_list.append(predictions)
        
        # print('y_pred: ', y_pred.shape)
        # print('y: ', y.shape)
        # y_pred = y_pred.view(-1).detach().cpu().tolist()
        # y = y.view(-1).detach().cpu().tolist()
        # y_pred = [[class_mapping[t] for t in s] for s in y_pred]
        
        # y_true = [[class_mapping[t] for t in s] for s in y_pred]
        # Map the integer labels back to NER tags
        # y_pred = [class_mapping[str(pred)] for pred in y_pred]
        # y_true = [class_mapping[str(pred)] for pred in y]

        

    precision, recall, f1 = evaluate(
        itertools.chain(*y_true_list),
        itertools.chain(*y_pred_list)
        )
    return precision, recall, f1, y_true_list, y_pred_list

def train_loop(config, device):
    """Implements training of the model.

    Arguments:
        config (dict): Contains configuration of the pipeline
        writer: tensorboardX writer object
        device: device on which to map the model and data
    """
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    reverse_class_mapping = {
        str(idx): cls_name for cls_name, idx in config["class_mapping"].items()
    }
    # Define dataloader hyper-parameters
    train_hyperparams = {
        "batch_size": config["batch_size"]["train"],
        "shuffle": True,
        "drop_last": True
    }
    valid_hyperparams = {
        "batch_size": config["batch_size"]["validation"],
        "shuffle": False,
        "drop_last": True
    }

    # Create dataloaders
    dataset = datasets.load_dataset("conll2003")
    train_set = CoNLLDataset(config, dataset["train"])
    valid_set = CoNLLDataset(config, dataset["validation"])
    train_loader = DataLoader(train_set, **train_hyperparams)
    valid_loader = DataLoader(valid_set, **valid_hyperparams)

    # Instantiate the model
    model = NERClassifier(config)
    model = model.to(device)

    # Load training configuration
    train_config = config["train_config"]
    learning_rate = train_config["learning_rate"]

    # Prepare the model optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["l2_penalty"]
    )

    # Weights used for Cross-Entropy loss
    # Calculated as log(1 / (class_count / train_samples))
    # @class_count: Number of tokens in the corpus per each class
    # @train_samples:  Total number of samples in the trains set
    class_w = train_config["class_w"]
    class_w = torch.tensor(class_w).to(device)
    class_w /= class_w.sum()

    train_step = 0
    start_time = time.strftime("%b-%d_%H-%M-%S")
    max_f1 = -1e+5
    model_ckp = ''
    for epoch in tqdm(range(train_config["num_of_epochs"])):
        print("Epoch:", epoch)
        model.train()
        loss_list = []
        for x, y, padding_mask in train_loader:
            train_step += 1
            x, y = x.to(device), y.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            y_pred = model(x, padding_mask)

            # Extract predictions and labels only for pre-padding tokens
            unpadded_mask = torch.logical_not(padding_mask)
            y = y[unpadded_mask]
            y_pred = y_pred[unpadded_mask]

            loss = F.cross_entropy(y_pred, y, weight=class_w)

            # Update model weights
            loss.backward()

            log_gradient_norm(model, train_step, "Before")
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["gradient_clipping"])
            log_gradient_norm(model, train_step, "Clipped")
            optimizer.step()
            loss_list.append(loss.item())
            # print("Train/Step-Loss", loss.item(), train_step)
            # print("Train/Learning-Rate", learning_rate, train_step)
        avg_loss = sum(loss_list) / len(loss_list)    
        print(f'Epoch {epoch}, Loss: {avg_loss}')
        with torch.no_grad():
            model.eval()
            evaluate_model(model, train_loader, device,
                           "Train", epoch, reverse_class_mapping)
            f1 = evaluate_model(model, valid_loader, device,
                           "Validation", epoch, reverse_class_mapping)
            model.train()
        if f1 > max_f1:
            save_checkpoint(model, start_time, epoch)
            max_f1 = f1
            model_ckp = f'{start_time}/model_{epoch}.pth'
    ckp = torch.load(f'checkpoints/{model_ckp}')
    model.load_state_dict(ckp)
    precision, recall, f1, y_true_list, y_pred_list = final_evaluate_model(model, valid_loader, device,
                           "Validation", reverse_class_mapping)
    print(f'precision={precision}, recall={recall}, f1={f1}')
    with open(f'checkpoints/prediction_{epoch}.json', 'w') as f:
        f.write(json.dumps({'predict': y_pred_list,
                            'true_label': y_true_list}))