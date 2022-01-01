import pymysql





mysql_conn = pymysql.connect(host='localhost',
                             user='hsucheng',
                             password='hsucheng',
                             port=3307)
mysql_cur = mysql_conn.cursor()
db_name = 'mysql_test'
sql = 'CREATE DATABASE {};'.format(db_name)
mysql_cur.execute(sql)
mysql_conn.commit()