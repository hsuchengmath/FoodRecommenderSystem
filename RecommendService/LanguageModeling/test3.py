




import torch.nn as nn
from transformers import BertModel
import torch



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return out


model = Model()
unfreeze_layer = ['layer.11', 'bert.pooler','out.']

for name, param in model.named_parameters():
    print(name, param.size())
print('---')
for name, param in model.named_parameters():
    param.requires_grad = False
    for layer in unfreeze_layer:
        if layer in name:
            param.requires_grad = True
            break

for name, param in model.named_parameters():
    print(name, param.size(), param.requires_grad)


print('-----')
#print(model(torch.tensor([[101]]) ))
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

inputs = tokenizer("肯德基好好吃", return_tensors="pt")
print(model(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask']))




print(inputs)
print(tokenizer.tokenize("肯德基好好吃"))

# generate pretrain input data


def GenerateInput(LeftSent=str, RightSent=str):
    #LeftSent = '肯德基好好吃'
    #RightSent = '麥當勞好好吃'

    head_token = ['CLS']
    tail_token = ['SEP']

    left_IDtoken = tokenizer.tokenize(LeftSent)
    right_IDtoken = tokenizer.tokenize(RightSent)

    pair_sent = head_token + left_IDtoken + tail_token + right_IDtoken + tail_token
    bert_id = tokenizer.convert_tokens_to_ids(pair_sent)
    bert_id_tesnor = torch.tensor(bert_id)

    segments_tensor = torch.tensor([0] * (len(left_IDtoken)+2) + [1] * (len(right_IDtoken)+1), dtype=torch.long)
    return bert_id_tesnor, segments_tensor

#print(model(input_ids=bert_id_tesnor, token_type_ids=segments_tensor, attention_mask=None))

#print(bert_id_tesnor)
#print(segments_tensor)
#print(left_IDtoken)
#print(right_IDtoken)

from torch.nn.utils.rnn import pad_sequence


data = [['肯德基好好吃', '肯德基'] ,['好吃', '吃']]



bert_id_tesnor_list, segments_tensor_list = [], []


for element in data:
    bert_id_tesnor, segments_tensor = GenerateInput(LeftSent=element[0], RightSent=element[1])
    bert_id_tesnor_list.append(bert_id_tesnor)
    segments_tensor_list.append(segments_tensor)



bert_id_tesnor = pad_sequence(bert_id_tesnor_list, batch_first=True)
segments_tensor = pad_sequence(segments_tensor_list, batch_first=True)
masks_tensors = torch.zeros(bert_id_tesnor.shape, dtype=torch.long)
masks_tensors = masks_tensors.masked_fill(bert_id_tesnor != 0, 1)

print(bert_id_tesnor)
print(segments_tensor)
print(masks_tensors)

print(model(input_ids=bert_id_tesnor, token_type_ids=segments_tensor, attention_mask=masks_tensors))

'''
https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L925
https://blog.csdn.net/HUSTHY/article/details/104006106
https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
BertModel <- 
'''