



from MenuDataPipeline.CollectMenuDataByStoreSN import CollectMenuDataByStoreSN
from MenuDataPipeline.CollectStoreSNByLatLon import CollectStoreSNByLatLon
from DataBaseService.InitDataBase import InitMongoDataBase
from DataBaseService.InsertData import InsertDataMongo



# collect menu data 
collection = []
longitude = 121.47232644568834
latitude = 25.030015106563397

StoreDataList = CollectStoreSNByLatLon(longitude=longitude, latitude=latitude ,limit=10)

for data in StoreDataList:
    StoreMenuList = CollectMenuDataByStoreSN(StoreSN=data['StoreSN'])
    collection += StoreMenuList




# insert the data to mongo
mongo_conn = InitMongoDataBase(db_name='MenuTEST')
InsertDataMongo(collection=collection, table_name='MenuTEST', mongo_conn=mongo_conn)