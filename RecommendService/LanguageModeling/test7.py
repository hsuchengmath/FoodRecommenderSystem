




import pickle



basic_bert_model_name = 'bert-base-chinese'
my_bert_model_name = 'AlbertHSU/ChineseFoodBert'
basic_exp_result_path = basic_bert_model_name.replace('/', '_').replace('-','_') + '@exp_result.pkl'
my_exp_result_path = my_bert_model_name.replace('/', '_').replace('-','_') + '@exp_result.pkl'



id2label = {'0':'台式', '1':'麵食', '2':'飲料', '3':'歐美', 
            '4':'便當', '5':'小吃', '6':'日式', '7':'甜點', 
            '8':'咖啡輕食', '9':'東南亞', '10':'健康餐', 
            '11':'火鍋', '12':'中式', '13':'早餐', '14':'炸雞'}



with open(basic_exp_result_path, "rb") as f:
    basic_data = pickle.load(f)
basic_FoodName2dPred_CS = basic_data['FoodName2dPred_CS']
with open(my_exp_result_path, "rb") as f:
    my_data = pickle.load(f)
my_FoodName2dPred_CS = my_data['FoodName2dPred_CS']

foodname_list = list(basic_FoodName2dPred_CS.keys())
index = 7
print('basic ({}) : '.format(foodname_list[index]), basic_FoodName2dPred_CS[foodname_list[index]])
print('---------------------')
print('my : ({}) '.format(foodname_list[index]), my_FoodName2dPred_CS[foodname_list[index]])