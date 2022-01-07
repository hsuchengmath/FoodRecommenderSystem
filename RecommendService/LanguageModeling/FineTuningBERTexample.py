











import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence





class BertLayer(nn.Module):
    def __init__(self, import_lm_name=str):
        super().__init__()
        self.bert_layer = BertModel.from_pretrained(import_lm_name)


    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        hidden_state = self.bert_layer(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return hidden_state


    def unfreeze_layer_action(self, unfreeze_layer=list):
        for name, param in self.named_parameters():
            param.requires_grad = False
            for layer in unfreeze_layer:
                if layer in name:
                    param.requires_grad = True
                    break



class BertForSequenceClassification(nn.Module):
    def __init__(self, config=dict):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = config['num_labels']

        # init bert layer
        self.bert_layer_module = BertLayer(import_lm_name=config['import_lm_name'])

        # unfreeze bert layer
        self.bert_layer_module.unfreeze_layer_action(unfreeze_layer=config['unfreeze_layer'])

        # init classfier layer
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.classifier = nn.Linear(config['hidden_size'], config['num_labels'])

        # init objective func
        self.loss_fct = nn.CrossEntropyLoss()


    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # bert layer
        hidden_state = self.bert_layer_module(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        # dropout layer
        hidden_state = self.dropout(hidden_state[1])
        
        # pred layer
        pred = self.classifier(hidden_state)
        
        # objective func
        loss = self.loss_fct(pred.view(-1, self.num_labels), labels.view(-1))
        return loss


 
def BertTokenization(LeftSent=str, RightSent=str, tokenizer='obj'):
    #LeftSent = '肯德基好好吃'
    #RightSent = '麥當勞好好吃'

    head_token = ['CLS']
    tail_token = ['SEP']

    left_IDtoken = tokenizer.tokenize(LeftSent)
    right_IDtoken = tokenizer.tokenize(RightSent)

    pair_sent = head_token + left_IDtoken + tail_token + right_IDtoken + tail_token
    bert_id = tokenizer.convert_tokens_to_ids(pair_sent)
    
    bert_id_tensor = torch.tensor(bert_id)
    segments_tensor = torch.tensor([0] * (len(left_IDtoken)+2) + [1] * (len(right_IDtoken)+1), dtype=torch.long)
    return bert_id_tensor, segments_tensor



def ConvertBertTokenizedBatch(batch_data=list, tokenizer='obj'):
    bert_id_tensor_list, segments_tensor_list, label_list = [], [], []
    for element in batch_data:
        bert_id_tensor, segments_tensor = BertTokenization(LeftSent=element[0], RightSent=element[1], tokenizer=tokenizer)
        bert_id_tensor_list.append(bert_id_tensor)
        segments_tensor_list.append(segments_tensor)
        label_list.append(element[2])
    bert_id_tensor = pad_sequence(bert_id_tensor_list, batch_first=True)
    segments_tensor = pad_sequence(segments_tensor_list, batch_first=True)
    masks_tensors = torch.zeros(bert_id_tensor.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(bert_id_tensor != 0, 1)
    label_tensor = torch.tensor([label_list])
    return bert_id_tensor, segments_tensor, masks_tensors, label_tensor






# init
config = \
    {
     'num_labels' : 2, 
     'import_lm_name' : "bert-base-chinese", 
     'unfreeze_layer' : ['layer.11', 'bert.pooler','out.'],
     'hidden_size' : 768,
     'hidden_dropout_prob' : 0.1
     }


# give example dataset
example_dataset = [['肯德基好好吃', '肯德基', 1] ,['好吃', '吃', 0]]


# init model
bert_classifier = BertForSequenceClassification(config=config)


# modeling
tokenizer = AutoTokenizer.from_pretrained(config['import_lm_name'])
bert_id_tensor, segments_tensor, masks_tensors, label_tensor = ConvertBertTokenizedBatch(batch_data=example_dataset, tokenizer=tokenizer)
loss = bert_classifier(input_ids=bert_id_tensor, token_type_ids=segments_tensor, attention_mask=masks_tensors, labels=label_tensor)
print('Loss : ', loss)




