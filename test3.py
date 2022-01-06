

import pymongo
StageMongodbUrl = 'mongodb://140.116.52.210:27018/'
collection_name = 'collection_test'
db_name = 'database_test'
myclient = pymongo.MongoClient(StageMongodbUrl) 
mydb = myclient[db_name] 
mycol = mydb[collection_name] 
mydict = {'MongodbUrl' : StageMongodbUrl, 'db_name' : db_name, 'db_name' : db_name, 'return' : 200}
mycol.insert_one(mydict)  
# test
for x in mycol.find():
   print(x)
 