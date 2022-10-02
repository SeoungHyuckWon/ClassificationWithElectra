import os
import pdb
import argparse
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm import tqdm, trange

from transformers import (
    ElectraForSequenceClassification,
    ElectraTokenizer,
    AutoConfig,
    AdamW
)

import pandas as pd


def make_id_file(task, tokenizer):
    def make_data_strings(file_name):
        data_strings = []

        with open(os.path.join(file_name), 'r', encoding='utf-8') as f:
            id_file_data = [tokenizer.encode(line.lower()) for line in f.readlines()]
        for item in id_file_data:
            data_strings.append(' '.join([str(k) for k in item]))
        return data_strings

    print('it will take some times...')
    train_pos = make_data_strings('sentiment.train.1')
    train_neg = make_data_strings('sentiment.train.0')
    dev_pos = make_data_strings('sentiment.dev.1')
    dev_neg = make_data_strings('sentiment.dev.0')

    print('make id file finished!')
    return train_pos, train_neg, dev_pos, dev_neg

# Data labeling(positive 1, negative 0)
class SentimentDataset(object):
    def __init__(self, tokenizer, pos, neg):
        self.tokenizer = tokenizer
        self.data = []
        self.label = []

        for pos_sent in pos:
            self.data += [self._cast_to_int(pos_sent.strip().split())]
            self.label += [[1]]
        for neg_sent in neg:
            self.data += [self._cast_to_int(neg_sent.strip().split())]
            self.label += [[0]]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample), np.array(self.label[index])

def collate_fn_style(samples):
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)
    sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1]

    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
                             batch_first=True)
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])
    labels = torch.tensor(np.stack(labels, axis=0)[sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids, labels

def compute_acc(predictions, target_labels):
    return (np.array(predictions) == np.array(target_labels)).mean()

def make_id_file_test(tokenizer, test_dataset):
    data_strings = []
    id_file_data = [tokenizer.encode(sent.lower()) for sent in test_dataset]
    for item in id_file_data:
        data_strings.append(' '.join([str(k) for k in item]))
    return data_strings

class SentimentTestDataset(object):
    def __init__(self, tokenizer, test):
        self.tokenizer = tokenizer
        self.data = []

        for sent in test:
            self.data += [self._cast_to_int(sent.strip().split())]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample)

# padding
def collate_fn_style_test(samples):
    input_ids = samples
    max_len = max(len(input_id) for input_id in input_ids)
    sorted_indices = range(len(input_ids))

    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
                             batch_first=True)
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids

def main():
    # by using ELECTRA
    tokenizer = BertTokenizer.from_pretrained('google/electra-base-generator')
    train_pos, train_neg, dev_pos, dev_neg = make_id_file('yelp', tokenizer)

    train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
    dev_dataset = SentimentDataset(tokenizer, dev_pos, dev_neg)

    train_batch_size=32
    eval_batch_size=64

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True, collate_fn=collate_fn_style,
                                               pin_memory=True, num_workers=2)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=eval_batch_size,
                                             shuffle=False, collate_fn=collate_fn_style,
                                             num_workers=2)

    # random seed
    random_seed=42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained('google/electra-base-generator')
    model.to(device)

    # learning rate, optimizer
    model.train()
    learning_rate = 5e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # training
    train_epoch = 3
    lowest_valid_loss = 9999.
    for epoch in range(train_epoch):
        with tqdm(train_loader, unit="batch") as tepoch:
            for iteration, (input_ids, attention_mask, token_type_ids, position_ids, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                position_ids = position_ids.to(device)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()

                output = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               labels=labels)

                loss = output.loss
                loss.backward()

                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                if iteration != 0 and iteration % int(len(train_loader) / 5) == 0:
                    # Evaluate the model five times per epoch
                    with torch.no_grad():
                        model.eval()
                        valid_losses = []
                        predictions = []
                        target_labels = []
                        for input_ids, attention_mask, token_type_ids, position_ids, labels in tqdm(dev_loader,
                                                                                                    desc='Eval',
                                                                                                    position=1,
                                                                                                    leave=None):
                            input_ids = input_ids.to(device)
                            attention_mask = attention_mask.to(device)
                            token_type_ids = token_type_ids.to(device)
                            position_ids = position_ids.to(device)
                            labels = labels.to(device, dtype=torch.long)

                            output = model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids,
                                           position_ids=position_ids,
                                           labels=labels)

                            logits = output.logits
                            loss = output.loss
                            valid_losses.append(loss.item())

                            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                            batch_labels = [int(example) for example in labels]

                            predictions += batch_predictions
                            target_labels += batch_labels

                    acc = compute_acc(predictions, target_labels)
                    valid_loss = sum(valid_losses) / len(valid_losses)
                    if lowest_valid_loss > valid_loss:
                        print('Acc for model which have lower valid loss: ', acc)
                        torch.save(model.state_dict(), "./pytorch_model.bin")

    # test
    test_df = pd.read_csv('test_no_label.csv')
    test_dataset = test_df['Id']
    test = make_id_file_test(tokenizer, test_dataset)
    test_dataset = SentimentTestDataset(tokenizer, test)

    test_batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                          shuffle=False, collate_fn=collate_fn_style_test,
                                          num_workers=2)
    with torch.no_grad():
        model.eval()
        predictions = []
        for input_ids, attention_mask, token_type_ids, position_ids in tqdm(test_loader,
                                                                            desc='Test',
                                                                            position=1,
                                                                            leave=None):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)

            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids)

            logits = output.logits
            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
            predictions += batch_predictions

        test_df['Category'] = predictions
        test_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
