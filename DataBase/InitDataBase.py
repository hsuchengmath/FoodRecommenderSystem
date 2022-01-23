


import pymysql
import pymongo
from DataBase.config import mysql_host
from DataBase.config import mysql_user
from DataBase.config import mysql_password
from DataBase.config import mysql_port
from DataBase.config import mongo_host




def InitMySQLDataBase(db_name=str):
    # connect mysql
    mysql_conn = pymysql.connect(host=mysql_host,
                                 user=mysql_user,
                                 password=mysql_password,
                                 port=mysql_port)
    mysql_cur = mysql_conn.cursor()
    
    # determine database name
    sql = 'USE {};'.format(db_name)
    mysql_cur.execute(sql)
    mysql_conn.commit()
    return mysql_cur, mysql_conn


def InitMongoDataBase(db_name=str):
    # connect mysql
    myclient = pymongo.MongoClient(mongo_host) 
    mongo_conn = myclient[db_name]
    return mongo_conn





