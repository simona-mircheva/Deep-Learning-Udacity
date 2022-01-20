import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

MAX_LEN = 64  # this is the max length of the sentence
batch_size=64
epochs=1

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_train_data_loader(batch_size):
    dataset = pd.read_csv(os.path.join("data", "train.csv"))
    sentences = dataset.sentence.values
    labels = dataset.label.values

    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)

    # pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader


def get_test_data_loader(test_batch_size):
    dataset = pd.read_csv(os.path.join("data", "test.csv"))
    sentences = dataset.sentence.values
    labels = dataset.label.values

    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)

    # pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=test_batch_size)

    return train_dataloader


def train(device):
    train_loader = get_train_data_loader(batch_size)
    test_loader = get_test_data_loader(batch_size)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8,  # args.adam_epsilon - default is 1e-8.
    )

    for epoch in range(1, epochs + 1):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            if step % 10  == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )

    test(model, test_loader, device)

def test(model, test_loader, device):
    model.eval()
    _, eval_accuracy = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy

    print("Test set: Accuracy: ", eval_accuracy/len(test_loader.dataset))

if __name__ == "__main__":
    df = pd.read_csv(
       "./cola_public/raw/in_domain_train.tsv",
       sep="\t",
       header=None,
       usecols=[1, 3],
       names=["label", "sentence"],
    )
    sentences = df.sentence.values
    labels = df.label.values

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df)
    train_df.to_csv("./data/train.csv", index=False)
    test_df.to_csv("./data/test.csv", index=False)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train(device)