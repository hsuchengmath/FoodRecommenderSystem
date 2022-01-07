
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs[1].shape) # torch.Size([1, 768])







import pymongo
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
 

# collect FoodNameList from mongoDB
print(1)
StageMongodbUrl = 'mongodb://140.116.52.210:27018/'
collection_name = 'MenuTEST'
db_name = 'MenuTEST'
myclient = pymongo.MongoClient(StageMongodbUrl)
mydb = myclient[db_name] 
mycol = mydb[collection_name] 

FoodNameList = set()
for x in mycol.find():
    FoodNameList.add(x['FoodName'])
FoodNameList = list(FoodNameList)
print(2)

# mapping foodname into bert-embedding
food_name2embedding = dict()
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
for i in tqdm(range(len(FoodNameList))):
    food_name = FoodNameList[i]
    inputs = tokenizer(food_name, return_tensors="pt")
    embedding = model(**inputs)[1]
    food_name2embedding[food_name] = embedding
print(3)

# build cos-sim between foodname-1, foodname-2 by F2F2S
FoodNameLongLeftList, FoodNameLongRightList = [], []
for i in tqdm(range(len(FoodNameList))):
    foodL = FoodNameList[i]
    for j in range(len(FoodNameList)):
        if i > j :
            foodR = FoodNameList[j]
            FoodNameLongLeftList.append(foodL)
            FoodNameLongRightList.append(foodR)
print(4)

FoodEmbLongLeftList = [food_name2embedding[food_name] for food_name in FoodNameLongLeftList]
FoodEmbLongRightList = [food_name2embedding[food_name] for food_name in FoodNameLongRightList]

FoodEmbLongLeft = torch.cat(FoodEmbLongLeftList, axis=0)
FoodEmbLongRight = torch.cat(FoodEmbLongRightList, axis=0)

print(5)

cosine_func = nn.CosineSimilarity(dim=1, eps=1e-6)
cos_score_list = cosine_func(FoodEmbLongLeft, FoodEmbLongRight).tolist()

print(6)

Food2Food2Score = dict()
for i in range(len(FoodNameLongLeftList)):
    foodL = FoodNameLongLeftList[i]
    foodR = FoodNameLongRightList[i]
    score = cos_score_list[i]
    if foodL not in Food2Food2Score:
        Food2Food2Score[foodL] = dict()
    if foodR not in Food2Food2Score:
        Food2Food2Score[foodR] = dict()
    Food2Food2Score[foodL][foodR] = score
    Food2Food2Score[foodR][foodL] = score

print(7)

# mapping food into sorted neighbor-food

Food2SimiliarFood = dict()
for food_name in FoodNameList:
    NeighborFood2Score = Food2Food2Score[food_name]
    neighbor_food_with_score = sorted(list(NeighborFood2Score.items()), reverse=True, key=lambda x:x[1])
    Food2SimiliarFood[food_name] = neighbor_food_with_score

print(8)

# store results
data = {'FoodNameList':FoodNameList, 'Food2SimiliarFood':Food2SimiliarFood, 'Food2Food2Score':Food2Food2Score}
with open('bert_result.pkl', "wb") as f:
    pickle.dump(data, f)



'''
https://chriskhanhtran.github.io/_posts/2019-12-25-bert-for-sentiment-analysis/
'''