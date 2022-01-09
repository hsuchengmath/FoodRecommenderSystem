






def QuerytDataMySQL(sql=str, mysql_cur='obj', mysql_conn='obj'):
    ''' 
    (1) : mysql_cur, mysql_conn : 'pymyssql-obj'
        The two object comes from InitSingleDataBase
    '''

    # query execute
    mysql_cur.execute(sql)
    mysql_conn.commit() 

    # convert into dataframe
    data_table = pd.DataFrame(data=mysql_cur.fetchall(), index = None, columns = mysql_cur.keys())

    return data_table
