

import json
import requests

try:
    from DataPipeline.config import CollectStoreSNByLatLon_url
    from DataPipeline.config import CollectStoreSNByLatLon_headers
except:
    from config import CollectStoreSNByLatLon_url
    from config import CollectStoreSNByLatLon_headers   




def CollectStoreSNByLatLon(longitude=float, latitude=float, limit=None):
    # collect organic data
    params = {'longitude' : longitude, 
             'latitude' : latitude,
             'language_id' : 6,
             'customer_type' : 'regular',
             'vertical' : 'restaurants',
             'configuration': 'Original',
             'country' : 'tw',
             'include' : 'characteristics',
             'offset' : 0,
             'sort' : 'distance_asc'}
    if limit is not None:
        params['limit'] = limit
    obj = requests.get(CollectStoreSNByLatLon_url, headers=CollectStoreSNByLatLon_headers, params=params)
    OrganicStoreData = json.loads(obj.text)

    # tidy up organic data
    StoreDataList = list()
    for data in OrganicStoreData['data']['items']:
        StoreData = dict()
        StoreData['StoreSN'] = data['id']
        StoreData['StoreAddress'] = data['address']
        StoreData['Budget'] = data['budget']
        StoreData['StoreName'] = data['name']
        StoreData['StoreRating'] = data['rating']
        StoreData['ReviewNumber'] = data['review_number']
        StoreData['StoreCuisines'] = data['cuisines']
        StoreDataList.append(StoreData)
    return StoreDataList

def example():
    '''
    print(data['data']['items'][0].keys())
    print('storeID : ',data['data']['items'][0]['id'])
    print('address : ', data['data']['items'][0]['address'])
    print('budget : ', data['data']['items'][0]['budget'])
    print('name : ', data['data']['items'][0]['name'])
    print('rating : ', data['data']['items'][0]['rating'])
    print('review_number : ', data['data']['items'][0]['review_number'])
    print('cuisines : ', data['data']['items'][0]['cuisines'])
    cuisines :  [{'id': 176, 'name': '甜點', 'url_key': 'dessert', 'main': True}]
    '''
    a = 0





longitude = 121.47232644568834
latitude = 25.030015106563397
if __name__ == '__main__':
    StoreDataList = CollectStoreSNByLatLon(longitude=longitude, latitude=latitude ,limit=10)
    print(StoreDataList)