

import requests




StoreSN = 124662
url = 'https://tw.fd-api.com/api/v5/vendors/{}'.format(str(StoreSN))


params = {'include' : 'menus',
          'language_id' : 6,
          'dynamic_pricing' : 0}
 


obj = requests.get(url, params=params)

import json
data = json.loads(obj.text)

print(data['data'].keys())




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
print('-----------------')
print(data['data']['menus'][0]['menu_categories'][0]['name']) #新品上市
print(data['data']['menus'][0]['menu_categories'][0]['description']) #限冰飲 | 冰量固定 | 建議半糖以上 | 夏日來杯涼涼的冰沙消暑吧
print(data['data']['menus'][0]['menu_categories'][0]['products'][0]['name']) #翡翠檸檬冰沙
print(data['data']['menus'][0]['menu_categories'][0]['products'][0]['file_path']) #https://images.deliveryhero.io/image/fd-tw/Products/33249112.jpg?width=%s
print(data['data']['menus'][0]['menu_categories'][0]['products'][0]['logo_path']) # same with file_path
print(data['data']['menus'][0]['menu_categories'][0]['products'][0]['product_variations'][0]['price'])
print(data['data']['menus'][0]['menu_categories'][0]['products'][0]['product_variations'][0]['container_price'])

