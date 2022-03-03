





# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from util import InputExample, InputFeatures


import os
import random 
import numpy as np
from tqdm import tqdm, trange

ENTAILMENT = 'entailment'
NON_ENTAILMENT = 'non_entailment'







class DNNC:
    def __init__(self,
                 internal_config: dict):
 
        self.internal_config = internal_config
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.device = torch.device("cpu")

        self.label_list = [ENTAILMENT, NON_ENTAILMENT]
        self.num_labels = len(self.label_list)

        bert_model = internal_config['bert_model']
        self.config = AutoConfig.from_pretrained(bert_model, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        path = internal_config['pretrain_model_path']
        if path is not None:
            state_dict = torch.load(path + '/pytorch_model.bin')
            self.model = AutoModelForSequenceClassification.from_pretrained(path, state_dict=state_dict, config=self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(bert_model, config=self.config)
        self.model.to(self.device)


    def save(self, 
             dir_path: str):
        # init parameter
        path = internal_config['intent_model_path']

        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model
        torch.save(model_to_save.state_dict(), '{}/pytorch_model.bin'.format(path))



    def convert_examples_to_features(self, examples, train):
        # init parameter
        max_seq_length = self.internal_config['max_seq_length']

        label_map = {label: i for i, label in enumerate(self.label_list)}
        is_roberta = True if "roberta" in self.config.architectures[0].lower() else False

        if train:
            label_distribution = torch.FloatTensor(len(label_map)).zero_()
        else:
            label_distribution = None

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)
            tokens_b = self.tokenizer.tokenize(example.text_b)

            # 互相傷害阿 by pop
            if is_roberta:
                truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
            else:
                truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
            segment_ids = [0] * len(tokens)

            if is_roberta:
                tokens_b = [self.tokenizer.sep_token] + tokens_b + [self.tokenizer.sep_token]
                segment_ids += [0] * len(tokens_b)
            else:
                tokens_b = tokens_b + [self.tokenizer.sep_token]
                segment_ids += [1] * len(tokens_b)
            tokens += tokens_b

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if example.label is None:
                label_id = -1
            else:
                label_id = label_map[example.label]

            if train:
                label_distribution[label_id] += 1.0

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

        if train:
            label_distribution = label_distribution / label_distribution.sum()
            return features, label_distribution
        else:
            return features



    def train(self, train_examples, dev_examples, file_path=None):
        # init parameter
        batch_size = self.internal_config['batch_size']
        gradient_accumulation_steps = self.internal_config['gradient_accumulation_steps']
        seed = self.internal_config['seed']
        epoch = self.internal_config['epoch']
        label_smoothing = self.internal_config['label_smoothing']
        max_grad_norm = self.internal_config['max_grad_norm']

        train_batch_size = int(batch_size / gradient_accumulation_steps)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        num_train_steps = int(len(train_examples) / train_batch_size / gradient_accumulation_steps * epoch)

        optimizer, scheduler = get_optimizer(self.model, num_train_steps, self.internal_config)

        best_dev_accuracy = -1.0

        train_features, label_distribution = self.convert_examples_to_features(train_examples, train = True)

        train_dataloader = get_train_dataloader(train_features, train_batch_size)


        self.model.zero_grad()
        self.model.train()
        for _ in trange(int(epoch), desc="Epoch"):

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                input_ids, input_mask, segment_ids, label_ids = process_train_batch(batch, self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                logits = outputs[0]
                loss = loss_with_label_smoothing(label_ids, logits, label_distribution, label_smoothing, self.device)

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()

            acc = self.evaluate(dev_examples)
            print('Validation Acc : ', acc)

            if acc > best_dev_accuracy and file_path is not None:
                best_dev_accuracy = acc
                self.save(file_path)
            
            self.model.train()



    def evaluate(self, eval_examples):
        # init parameter
        batch_size = self.internal_config['batch_size']

        if len(eval_examples) == 0:
            return None

        eval_features = self.convert_examples_to_features(eval_examples, train = False)
        eval_dataloader = get_eval_dataloader(eval_features, batch_size)
        
        self.model.eval()
        eval_accuracy = 0
        nb_eval_examples = 0

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)

        eval_accuracy = eval_accuracy / nb_eval_examples
        return eval_accuracy



    def predict(self,
                data):

        self.model.eval()

        input = [InputExample(premise, hypothesis) for (premise, hypothesis) in data]

        eval_features = self.convert_examples_to_features(input, train = False)
        input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        max_len = input_mask.sum(dim=1).max().item()
        input_ids = input_ids[:, :max_len]
        input_mask = input_mask[:, :max_len]
        segment_ids = segment_ids[:, :max_len]

        CHUNK = 500
        EXAMPLE_NUM = input_ids.size(0)
        labels = []
        probs = None
        start_index = 0

        while start_index < EXAMPLE_NUM:
            end_index = min(start_index+CHUNK, EXAMPLE_NUM)
            
            input_ids_ = input_ids[start_index:end_index, :].to(self.device)
            input_mask_ = input_mask[start_index:end_index, :].to(self.device)
            segment_ids_ = segment_ids[start_index:end_index, :].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids_, attention_mask=input_mask_, token_type_ids=segment_ids_)
                logits = outputs[0]
                probs_ = torch.softmax(logits, dim=1)

            probs_ = probs_.detach().cpu()
            if probs is None:
                probs = probs_
            else:
                probs = torch.cat((probs, probs_), dim = 0)
            labels += [self.label_list[torch.max(probs_[i], dim=0)[1].item()] for i in range(probs_.size(0))]
            start_index = end_index

        assert len(labels) == EXAMPLE_NUM
        assert probs.size(0) == EXAMPLE_NUM
            
        return labels, probs




def get_optimizer(model, t_total, internal_config):
    # init parameter
    learning_rate = internal_config['learning_rate']
    adam_epsilon = internal_config['adam_epsilon']
    warmup_proportion = internal_config['warmup_proportion']

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * warmup_proportion),
                                                num_training_steps=t_total)
    
    return optimizer, scheduler



def truncate_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



def get_train_dataloader(train_features, train_batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    return train_dataloader



def process_train_batch(batch, device):
    input_mask = batch[1]
    batch_max_len = input_mask.sum(dim=1).max().item()

    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch
    input_ids = input_ids[:, :batch_max_len]
    input_mask = input_mask[:, :batch_max_len]
    segment_ids = segment_ids[:, :batch_max_len]
    return input_ids, input_mask, segment_ids, label_ids



def loss_with_label_smoothing(label_ids, logits, label_distribution, coeff, device):
    # label smoothing
    label_ids = label_ids.cpu()
    target_distribution = torch.FloatTensor(logits.size()).zero_()
    for i in range(label_ids.size(0)):
        target_distribution[i, label_ids[i]] = 1.0
    target_distribution = coeff * label_distribution.unsqueeze(0) + (1.0 - coeff) * target_distribution
    target_distribution = target_distribution.to(device)

    # KL-div loss
    prediction = torch.log(torch.softmax(logits, dim=1))
    loss = F.kl_div(prediction, target_distribution, reduction='mean')
    return loss



def get_eval_dataloader(eval_features, eval_batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    return eval_dataloader



def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)




class IntentPredictor:
    def __init__(self,
                 tasks = None):

        self.tasks = tasks

    def predict_intent(self,
                       input: str):
        raise NotImplementedError



class DnncIntentPredictor(IntentPredictor):
    def __init__(self,
                 model,
                 tasks = None):
        
        super().__init__(tasks)
        
        self.model = model

    def predict_intent(self,
                       input: str):

        nli_input = []
        for t in self.tasks:
            for e in t['examples']:
                nli_input.append((input, e))

        assert len(nli_input) > 0

        results = self.model.predict(nli_input)
        maxScore, maxIndex = results[1][:, 0].max(dim = 0)

        maxScore = maxScore.item()
        maxIndex = maxIndex.item()

        index = -1
        for t in self.tasks:
            for e in t['examples']:
                index += 1

                if index == maxIndex:
                    intent = t['task']
                    matched_example = e

        return intent, maxScore, matched_example








# intent_predictor = DnncIntentPredictor(model, sampled_tasks[j])

#         in_domain_preds = []
#         oos_preds = []

#         for e in tqdm(dev_examples, desc = 'Intent examples'):
#             pred, conf, matched_example = intent_predictor.predict_intent(e.text)
#             in_domain_preds.append((conf, pred))

#             if args.save_model_path and args.do_predict:
#                 if not trial_stats_preds[e.label]:
#                     trial_stats_preds[e.label] = []

#                 single_pred = {}
#                 single_pred['gold_example'] = e.text
#                 single_pred['match_example'] = matched_example
#                 single_pred['gold_label'] = e.label
#                 single_pred['pred_label'] = pred
#                 single_pred['conf'] = conf
#                 trial_stats_preds[e.label].append(single_pred)

#         for e in tqdm(oos_dev_examples, desc = 'OOS examples'):
#             pred, conf, matched_example = intent_predictor.predict_intent(e.text)
#             oos_preds.append((conf, pred))

#             if args.save_model_path and args.do_predict:
#                 if not trial_stats_preds[e.label]:
#                     trial_stats_preds[e.label] = []

#                 single_pred = {}
#                 single_pred['gold_example'] = e.text
#                 single_pred['match_example'] = matched_example
#                 single_pred['gold_label'] = e.label
#                 single_pred['pred_label'] = pred
#                 single_pred['conf'] = conf
#                 trial_stats_preds[e.label].append(single_pred)

#         if args.save_model_path and args.do_predict:
#             stats_lists_preds[j] = trial_stats_preds

#         in_acc = calc_in_acc(dev_examples, in_domain_preds, THRESHOLDS)
#         oos_recall = calc_oos_recall(oos_preds, THRESHOLDS)
#         oos_prec = calc_oos_precision(in_domain_preds, oos_preds, THRESHOLDS)
#         oos_f1 = calc_oos_f1(oos_recall, oos_prec)

#         print_results(THRESHOLDS, in_acc, oos_recall, oos_prec, oos_f1)
