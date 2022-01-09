






def InsertDataMySQL(collection=list, table_name=str, mysql_cur='obj', mysql_conn='obj'):
    '''
    (1) : collection : list
        ex. [{'CustomerName' : 'Cardinal', 'ContactName' : 'Tom B. Erichsen'}, ....]
    
    (2) : mysql_cur, mysql_conn : 'pymyssql-obj'
        The two object comes from InitSingleDataBase
    '''

    # insert collection data to target table_name

    sql = '''
    INSERT INTO {} ({})
    VALUES ({});
    '''
    column_name = list(collection[0].keys())
    column_name_str = ','.join(["'"+col+"'" for col in column_name])
    
    token_str = ','.join(['%s' for col in column_name])
    
    sql = sql.format(table_name, column_name_str, token_str)
    
    records_to_insert = [[data[col] for col in column_name] for data in collection]
    
    mysql_cur.executemany(sql, records_to_insert)
    
    mysql_conn.commit()



def InsertDataMongo(collection=list, table_name=str, mongo_conn='obj'):
    '''
    (1) : collection : list
        ex. [{'CustomerName' : 'Cardinal', 'ContactName' : 'Tom B. Erichsen'}, ....]
    '''
    mycol = mongo_conn[table_name] 
    for data in collection:
        mycol.insert_one(data) 

