

import json
import requests

from MenuDataPipeline.config import CollectMenuDataByStoreSN_url







def CollectMenuDataByStoreSN(StoreSN=int):
    # collect organic data
    params = {'include' : 'menus',
              'language_id' : 6,
              'dynamic_pricing' : 0}
    obj = requests.get(CollectMenuDataByStoreSN_url, params=params)
    OrganicMenuData = json.loads(obj.text)

    # tidy up organic data
    MenuData = dict()
    MenuData['StoreSN'] = OrganicMenuData['data']['id']
    MenuData['StoreAddress'] = OrganicMenuData['data']['address']
    MenuData['Budget'] = OrganicMenuData['data']['budget']
    MenuData['StoreName'] = OrganicMenuData['data']['name']
    MenuData['StoreRating'] = OrganicMenuData['data']['rating']
    MenuData['ReviewNumber'] = OrganicMenuData['data']['review_number']
    MenuData['StoreCuisines'] = OrganicMenuData['data']['cuisines']
    MenuData['OpeningTime'] = OrganicMenuData['menus'][0]['opening_time']
    MenuData['ClosingTime'] = OrganicMenuData['menus'][0]['closing_time']
    StoreMenuList = list()
    menu_categories = OrganicMenuData['menus'][0]['menu_categories']
    for category_data in menu_categories:
        MenuCategoryName = category_data['name']
        MenuCategoryDescription = category_data['description']
        MenuCategoryProductsList = category_data['products']
        for food_data in MenuCategoryProductsList:
            FoodName = food_data['name']
            FoodImgUrl1 = food_data['file_path']
            FoodImgUrl2 = food_data['logo_path']
            FoodPrice = food_data['product_variations'][0]['price']
            FoodContainerPrice = food_data['product_variations'][0]['container_price']
            FoodData = dict()
            FoodData['MenuData'] = MenuData
            FoodData['MenuCategoryName'] = MenuCategoryName
            FoodData['MenuCategoryDescription'] = MenuCategoryDescription
            FoodData['FoodName'] = FoodName
            FoodData['FoodImgUrl1'] = FoodImgUrl1
            FoodData['FoodImgUrl2'] = FoodImgUrl2
            FoodData['FoodPrice'] = FoodPrice
            FoodData['FoodContainerPrice'] = FoodContainerPrice
            StoreMenuList.append(FoodData)
    return StoreMenuList


'''
print(data['data']['id'])
print(data['data']['address'])
print(data['data']['budget'])
print(data['data']['name'])
print(data['data']['rating'])
print(data['data']['review_number'])
print(data['data']['cuisines'])
print(data['data']['distance'])
print(data['data']['menus'][0]['id'])#180409
print(data['data']['menus'][0]['opening_time'])#00:00:00
print(data['data']['menus'][0]['closing_time'])#23:59:00
print(data['data']['menus'][0]['menu_categories'][1]['name']) #新品上市
print(data['data']['menus'][0]['menu_categories'][1]['description']) #限冰飲 | 冰量固定 | 建議半糖以上 | 夏日來杯涼涼的冰沙消暑吧
print(data['data']['menus'][0]['menu_categories'][1]['products'][0]['name']) #翡翠檸檬冰沙
print(data['data']['menus'][0]['menu_categories'][1]['products'][0]['file_path']) #https://images.deliveryhero.io/image/fd-tw/Products/33249112.jpg?width=%s
print(data['data']['menus'][0]['menu_categories'][1]['products'][0]['logo_path']) # same with file_path
print(data['data']['menus'][0]['menu_categories'][1]['products'][0]['product_variations'][0]['price'])
print(data['data']['menus'][0]['menu_categories'][1]['products'][0]['product_variations'][0]['container_price'])
'''

if __name__ == '__main__':
    StoreMenuList = CollectMenuDataByStoreSN(StoreSN=124662)
    print(StoreMenuList)