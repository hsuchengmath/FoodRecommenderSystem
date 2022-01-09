

import json
import requests


try:
    from MenuDataPipeline.config import CollectMenuDataByStoreSN_url
except:
    from config import CollectMenuDataByStoreSN_url







def CollectMenuDataByStoreSN(StoreSN=int):
    # collect organic data
    params = {'include' : 'menus',
              'language_id' : 6,
              'dynamic_pricing' : 0}
    CollectMenuDataByStoreSN_url_withSN = CollectMenuDataByStoreSN_url.format(str(StoreSN))
    obj = requests.get(CollectMenuDataByStoreSN_url_withSN, params=params)
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
    MenuData['OpeningTime'] = OrganicMenuData['data']['menus'][0]['opening_time']
    MenuData['ClosingTime'] = OrganicMenuData['data']['menus'][0]['closing_time']
    StoreMenuList = []
    menu_categories = OrganicMenuData['data']['menus'][0]['menu_categories']
    for category_data in menu_categories:
        MenuCategoryProductsList = category_data['products']
        for food_data in MenuCategoryProductsList:
            FoodData = dict()
            FoodData['MenuData'] = MenuData
            FoodData['MenuCategoryName'] = category_data['name']
            FoodData['MenuCategoryDescription'] = category_data['description']
            FoodData['FoodName'] = food_data['name']
            FoodData['FoodImgUrl1'] = food_data['file_path']
            FoodData['FoodImgUrl2'] = food_data['logo_path']
            FoodData['FoodPrice'] = food_data['product_variations'][0]['price']
            FoodData['FoodContainerPrice'] = food_data['product_variations'][0]['container_price']
            StoreMenuList.append(FoodData)
    return StoreMenuList



def example():
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
    a = 0






if __name__ == '__main__':
    StoreMenuList = CollectMenuDataByStoreSN(StoreSN=13851)
    print(StoreMenuList[0])