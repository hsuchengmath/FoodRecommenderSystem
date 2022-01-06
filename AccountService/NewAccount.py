


import pandas as pd


from DataBaseService.InitDataBase import InitSingleDataBase
from DataBaseService.InsertData import InsertDataMySQL
from DataBaseService.QuerytDataMySQL import QuerytDataMySQL
from DataBaseService.config import AccountInfo
from DataBaseService.config import account_password




def InsertNewAccount(account=str, password=str, root=False):
    # init database-table
    mysql_cur, mysql_conn = InitSingleDataBase(db_name=AccountInfo)


    # check the account have used or not
    forward = True
    QuerytDataMySQL(sql=str, mysql_cur=mysql_cur, mysql_conn=mysql_conn)

    # check root account or not
    if root is True:
        root = 1
    else:
        root = 0
    
    # if new account, we insert the account to database 
    if forward is True:
        collection = [{'account':account, 'password':password, 'root':root, 'modified_date':'2021-12-12 00:00:30'}]
        InsertDataMySQL(collection=collection, table_name=account_password, mysql_cur=mysql_cur, mysql_conn=mysql_conn)
        return 200
    else:
        return 888




