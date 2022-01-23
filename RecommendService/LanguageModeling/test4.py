



import pymongo
StageMongodbUrl = 'mongodb://140.116.52.210:27018/'
collection_name = 'PttCorpusTEST'
db_name = 'PttCorpusTEST'
collection_name = 'PttCorpus'
db_name = 'PttCorpus-Food'
myclient = pymongo.MongoClient(StageMongodbUrl) 
mydb = myclient[db_name] 
mycol = mydb[collection_name] 
#mydict = {'MongodbUrl' : StageMongodbUrl, 'db_name' : db_name, 'db_name' : db_name, 'return' : 200}
#mycol.insert_one(mydict)  
# test


ArticleList = []
for x in mycol.find():
    url = 'https://www.ptt.cc/bbs/Food/{}.html'.format(x['article_id'])
    data = {'title':x['title'], 'content':x['content'], 'url' : url}
    ArticleList.append(data)
print(ArticleList[100])
print(ArticleList[100]['url'])
print(len(ArticleList))



