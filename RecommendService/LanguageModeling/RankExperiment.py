


import torch
import pickle
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from EvaluationUtil import precision_at_k, recall_at_k, ndcg_k




# init
label_data_path = 'LabelData/FoodNameLabel.pkl'
bert_model_name = 'bert-base-chinese'#'AlbertHSU/ChineseFoodBert'

id2label = {'0':'台式', '1':'麵食', '2':'飲料', '3':'歐美', 
            '4':'便當', '5':'小吃', '6':'日式', '7':'甜點', 
            '8':'咖啡輕食', '9':'東南亞', '10':'健康餐', 
            '11':'火鍋', '12':'中式', '13':'早餐', '14':'炸雞'}
LabelNameList = list(id2label.values())
cosine_func = nn.CosineSimilarity(dim=1, eps=1e-6)
TopK = 5
exp_result_path = bert_model_name.replace('/', '_').replace('-','_') + '@exp_result.pkl'


# load FoodNameLabel data
with open(label_data_path, "rb") as f:
    data = pickle.load(f)
DataList = data['Data']
DataList_nonempty = [element for element in DataList if element['Label'] is not None and element['Label'][0] != 'X']



# import bert model
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)


#labelname embedding
LabelName2Embedding = dict()
for labelname in LabelNameList:
    inputs = tokenizer(labelname, return_tensors="pt")
    embedding = model(**inputs)[1]
    LabelName2Embedding[labelname] = embedding
LabelNameEmbeddingList = [LabelName2Embedding[labelname] for labelname in LabelNameList]
LabelNameEmbLongLeft = torch.cat(LabelNameEmbeddingList, axis=0)


 
# calculate cosine-similiar(CS) order
FoodName2dPred_CS = dict()
for element in DataList_nonempty:
    FoodName = element['FoodName']
    GroundTrueLabel = element['Label']
    # Foodname embedding
    inputs = tokenizer(FoodName, return_tensors="pt")
    embedding = model(**inputs)[1] 
    # convet to long embedding
    FoodNameEmbLongList = [embedding for _ in range(len(LabelNameList))]
    FoodNameEmbLong = torch.cat(FoodNameEmbLongList, axis=0)
    # calculate cs and sort
    cos_score_list = cosine_func(FoodNameEmbLong, LabelNameEmbLongLeft).tolist()
    LabelName_with_CS = [[LabelNameList[i],cos_score_list[i]] for i in range(len(LabelNameList))]
    LabelName_with_CS = sorted(LabelName_with_CS, reverse=True, key= lambda x:x[1])
    # append
    FoodName2dPred_CS[FoodName] = LabelName_with_CS




# evaluation 
actual, predicted = [], []

for element in DataList_nonempty:
    FoodName = element['FoodName']
    GroundTrueLabel = element['Label']
    if GroundTrueLabel[0] != 'X':
        GroundTrueLabelName = [id2label[id_] for id_ in GroundTrueLabel]
        Pred_CS = FoodName2dPred_CS[FoodName]
        Pred = [element[0] for element in Pred_CS]
        actual.append(GroundTrueLabelName)
        predicted.append(Pred)

print('Precision@K={} : '.format(str(TopK)), precision_at_k(actual=actual, predicted=predicted, topk=TopK))
print('Recall@K={} : '.format(str(TopK)), recall_at_k(actual=actual, predicted=predicted, topk=TopK))
print('NDCG@K={} : '.format(str(TopK)), ndcg_k(actual=actual, predicted=predicted, topk=TopK))


# save exp result to pickle
data = {'RankMetric' : {
                        'Precision' : precision_at_k(actual=actual, predicted=predicted, topk=TopK),
                        'Recall' : recall_at_k(actual=actual, predicted=predicted, topk=TopK),
                        'NDCG' : ndcg_k(actual=actual, predicted=predicted, topk=TopK)
                       },
        'DataList_nonempty' : DataList_nonempty,
        'FoodName2dPred_CS' : FoodName2dPred_CS,
        'id2label' : id2label}
with open(exp_result_path, "wb") as f:
    pickle.dump(data, f)
