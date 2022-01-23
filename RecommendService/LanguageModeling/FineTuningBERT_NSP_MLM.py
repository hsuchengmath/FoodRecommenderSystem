











import torch
import torch.nn as nn
import random
from torch.nn.functional import one_hot as convert_id_to_onehot
from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.nn.functional import gelu as GeluActMLMivationLayer




class BertLayer(nn.Module):
    def __init__(self, import_lm_name=str):
        super().__init__()
        self.bert_layer = BertModel.from_pretrained(import_lm_name)
        self.vocab_num = self.bert_layer.config.vocab_size


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
        # init bert layer
        self.bert_layer_module = BertLayer(import_lm_name=config['import_lm_name']).to(config['device'])

        # init parameter
        self.NSP_num_labels = config['NSP_num_labels']
        self.MLM_num_labels = self.bert_layer_module.vocab_num
        self.vocab_num = self.bert_layer_module.vocab_num


        # unfreeze bert layer
        self.bert_layer_module.unfreeze_layer_action(unfreeze_layer=config['unfreeze_layer'])

        # dropout layer 
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

        # init NSP classfier layer
        self.classifier_NSP = nn.Linear(config['hidden_size'], config['NSP_num_labels'])
        
        # init MLM classfier layer
        self.decoder_MLM = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.classifier_MLM = nn.Linear(config['hidden_size'], self.MLM_num_labels)

        # init objective func
        self.loss_fct = nn.CrossEntropyLoss()




    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, NSPlabel=None, MLMlabel=None):
        # bert layer
        hidden_state = self.bert_layer_module(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        # dropout layer
        hidden_state_NSP = self.dropout(hidden_state[1])
        hidden_state_MLM = self.dropout(hidden_state[0])

        # pred layer (NSP)
        pred_NSP = self.classifier_NSP(hidden_state_NSP)

        # pred layer (MLM)
        hidden_state_MLM = self.decoder_MLM(hidden_state_MLM)
        hidden_state_MLM = GeluActMLMivationLayer(hidden_state_MLM)
        hidden_state_MLM = self.LayerNorm(hidden_state_MLM)
        pred_MLM = self.classifier_MLM(hidden_state_MLM)
        
        # objective func 
        print('pred_MLM : ',pred_MLM.shape)
        print('MLMlabel : ',MLMlabel.shape)
        loss_NSP = self.loss_fct(pred_NSP.view(-1, self.NSP_num_labels), NSPlabel.view(-1))
        loss_MLM = self.loss_fct(pred_MLM, MLMlabel)

        return loss_NSP + loss_MLM, loss_NSP, loss_MLM
        


 
def BertTokenization(LeftSent=str, RightSent=str, MaskLabel=list, tokenizer='obj', vocab_num=int):
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

    MaskLabel = random.sample([i+1 for i in range(len(left_IDtoken))], int(len(left_IDtoken)*0.15)+1)


    onehot_mask_tensor = torch.sum(convert_id_to_onehot(torch.tensor(MaskLabel), num_classes=vocab_num), axis=0)

    return bert_id_tensor, segments_tensor, onehot_mask_tensor



def ConvertBertTokenizedBatch(batch_data=list, tokenizer='obj', config=dict, vocab_num=int):
    bert_id_tensor_list, token_type_list, NSP_label_list, MLM_label_list = [], [], [], []
    for element in batch_data:
        bert_id_tensor, token_type_tensor, onehot_mask_tensor = \
            BertTokenization(LeftSent=element[0], RightSent=element[1], MaskLabel=element[3], tokenizer=tokenizer, vocab_num=vocab_num)
        if bert_id_tensor.shape[0] < config['max_charator_length']:
            bert_id_tensor_list.append(bert_id_tensor)
            token_type_list.append(token_type_tensor)
            NSP_label_list.append(element[2])
            MLM_label_list.append(onehot_mask_tensor)
    if len(bert_id_tensor_list) != 0:
        bert_id_tensor = pad_sequence(bert_id_tensor_list, batch_first=True).to(config['device'])
        token_type_tensor = pad_sequence(token_type_list, batch_first=True).to(config['device'])
        MLMlabel_tensor = pad_sequence(MLM_label_list, batch_first=True).to(config['device'])
        print('bert_id_tensor : ', bert_id_tensor.shape)
        print('MLMlabel_tensor : ', MLMlabel_tensor.shape)
        masks_tensors = torch.zeros(bert_id_tensor.shape, dtype=torch.long).to(config['device'])
        masks_tensors = masks_tensors.masked_fill(bert_id_tensor != 0, 1).to(config['device'])
        NSPlabel_tensor = torch.tensor([NSP_label_list]).to(config['device'])
        return bert_id_tensor, token_type_tensor, masks_tensors, NSPlabel_tensor, MLMlabel_tensor
    else:
        return None, None, None, None

 




# init
config = \
    {
     'NSP_num_labels' : 2, 
     'import_lm_name' : "bert-base-chinese", 
     'unfreeze_layer' : ['layer.11', 'bert.pooler','out.'],
     'hidden_size' : 768,
     'hidden_dropout_prob' : 0.1,
     'mask_prob' : 0.15, 
     'epoch' : 20,
     'learning_rate': 1e-5,
     'layer_norm_eps' : 1e-12,
     'batch_size' : 128,
     'max_charator_length' : 512,
     'db_name' : 'PttCorpus-Food',
     'collection_name' : 'PttCorpus',
     'BertName' : 'ChineseFoodBert',
     'device' : "cuda:0" if torch.cuda.is_available() else "cpu"
    }


# give example dataset
example_dataset = [['肯德基好好吃', '肯德基', 1] ,['好吃', '吃', 0]]

from DataProcessTEST import DataProcessTEST
train_data = DataProcessTEST(config=config)
 

# init model
bert_classifier = BertForSequenceClassification(config=config).to(config['device'])
vocab_num = bert_classifier.vocab_num


# modeling
optimizer = torch.optim.Adam(bert_classifier.parameters(), lr=config['learning_rate'])
tokenizer = AutoTokenizer.from_pretrained(config['import_lm_name'])
batfch_num = int(len(train_data) / config['batch_size']) + 1

for epoch in range(config['epoch']):
    print('epoch : ',epoch)
    total_loss, total_loss_NSP, total_loss_MLM = 0, 0, 0
    for i in tqdm(range(batfch_num)):
        batch_data = train_data[i * config['batch_size'] : (i+1) * config['batch_size']]

        # convert organic text into token_id
        bert_id_tensor, token_type_tensor, masks_tensors, NSPlabel_tensor, MLMlabel_tensor = \
            ConvertBertTokenizedBatch(batch_data=batch_data, tokenizer=tokenizer, config=config, vocab_num=vocab_num)

        if bert_id_tensor is not None:
            # init optimization
            optimizer.zero_grad()

            # loss
            loss, loss_NSP, loss_MLM = bert_classifier(input_ids=bert_id_tensor, 
                                                       token_type_ids=token_type_tensor, 
                                                       attention_mask=masks_tensors, 
                                                       NSPlabel=NSPlabel_tensor,
                                                       MLMlabel=MLMlabel_tensor)
            total_loss += loss
            total_loss_NSP += loss_NSP
            total_loss_MLM += loss_MLM

            # backward
            loss.backward()
            optimizer.step()
    print('total_loss : ', total_loss)
    print('total_loss_NSP : ', total_loss_NSP)
    print('total_loss_MLM : ', total_loss_MLM)
    print('=======================')


#bert_classifier.bert_layer_module.bert_layer.save_pretrained(config['BertName'])

'''
https://zhuanlan.zhihu.com/p/390826470

----------------


git clone https://huggingface.co/AlbertHSU/ChineseFoodBert
copy musthave model-file to target-file
git lfs install
git add .
git lfs status
git commit -m "add basic bert-model file"
git push (account, password[token])

----
how to install git-lfs =>
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
'''