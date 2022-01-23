





from DataPipeline.CollectPttCorpusByBoardNames import CrawlPttData



dat = CrawlPttData(Board_Name ='Food',page_num= 50, column_eng=True)


import pymongo
StageMongodbUrl = 'mongodb://140.116.52.210:27018/'
collection_name = 'PttCorpus'
db_name = 'PttCorpus-Food'
myclient = pymongo.MongoClient(StageMongodbUrl) 
mydb = myclient[db_name] 
mycol = mydb[collection_name] 


collection = dat.to_dict('records')

for data in collection:
    mycol.insert_one(data)  



