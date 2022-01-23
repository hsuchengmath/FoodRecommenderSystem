

import pymongo
import pickle



# parameter
hand_craft_annotation = True
show_all_foodname = True
modify_by_foodname = False
label_file_name = 'FoodNameLabel'
id2label = {'0':'台式', '1':'麵食', '2':'飲料', '3':'歐美', 
            '4':'便當', '5':'小吃', '6':'日式', '7':'甜點', 
            '8':'咖啡輕食', '9':'東南亞', '10':'健康餐', 
            '11':'火鍋', '12':'中式', '13':'早餐', '14':'炸雞'}


    

def HandCraftAnnotation(element=dict):
    print('FoodName : ', element['FoodName'])
    print('LabelCategoryName : ', id2label)
    print('---------------------------------------')
    InputLabel = input('Please Input Label : ')
    InputLabelList = InputLabel.split(',')
    print('Your input : ', InputLabelList)
    return InputLabelList




# collect FoodName from db
if hand_craft_annotation is False:
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

# save all data to pickle and format by {FoodName:xxx, label:None}
if hand_craft_annotation is False:
    data = {'Data':[]}
    for food_name in FoodNameList:
        data['Data'].append({'FoodName' : food_name, 'Label' : None})
    with open(label_file_name+".pkl", "wb") as f:
        pickle.dump(data, f)

# show all food name
if show_all_foodname is True:
    with open(label_file_name+".pkl", "rb") as f:
        data = pickle.load(f)   
    DataList = data['Data']
    all_foodname = []
    for element in DataList:
        if element['Label'] is not None:
            all_foodname.append(element['FoodName'])
    print('all_foodname : ', all_foodname)
    print('finish rate : ', len(all_foodname)/len(DataList))
    print('all_foodname len : ', len(all_foodname))
        
# modify 
if modify_by_foodname is True:
    modified_food_name = input('Please input modified food name : ')
    print('Your input : ', modified_food_name)

    with open(label_file_name+".pkl", "rb") as f:
        data = pickle.load(f)   
    DataList = data['Data']
    for element in DataList:
        if element['FoodName'] == modified_food_name and element['Label'] is not None:
            print('The label of {} is : '.format(modified_food_name), \
                  [id2label[id_] for id_ in element['Label'] if id_ in id2label])
            

# HAND-CRAFT
if hand_craft_annotation is True and show_all_foodname is False and modify_by_foodname is False:
    with open(label_file_name+".pkl", "rb") as f:
        data = pickle.load(f)
    DataList = data['Data']
    for i in range(len(DataList)):
        element = DataList[i]

        if element['Label'] is None:
            forward = True
            while forward is True:
                InputLabelList = HandCraftAnnotation(element=element)
                check = input('if it is well, then directly input ENTER (withoud any something)')
                if len(check) == 0:
                    forward = False
            element['Label'] = InputLabelList
            DataList[i] = element
            check = input('if it keeps going now, then directly input ENTER, ELSE input NO!!')
            if check == 'NO':
                break
    data['Data'] = DataList
    with open(label_file_name+".pkl", "wb") as f:
        pickle.dump(data, f)
'''
if "X" it means the data will be removed!!
'''