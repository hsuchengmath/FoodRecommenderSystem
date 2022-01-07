






    
import pickle
with open('bert_result.pkl', "rb") as f:
    output = pickle.load(f)
print(output.keys())
Food2SimiliarFood = output['Food2SimiliarFood']
FoodNameList = output['FoodNameList']
print(FoodNameList[:10])
print(Food2SimiliarFood[FoodNameList[2]])