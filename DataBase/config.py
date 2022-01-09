



# mysql init setting
mysql_host = 'localhost'
mysql_user = 'hsucheng'
mysql_password = 'hsucheng'
mysql_port = 3307

# mysql database name
AccountInfo = 'AccountInfo'
account_password = 'account_password'

# mongo init setting
mongo_host = 'mongodb://140.116.52.210:27018/'



docker run -d  -p 9010:9010 --name portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer