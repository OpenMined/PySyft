HOST = "localhost"
PORT = 5959
# name of the Table Database
TABLE_DB_KEY = "syft-table-db"
# name of the DHT Key in the table Database
DHT_KEY = "syft-dht-key"
# name of the DHT Key Credentials in the table Database
# Credentials refer to the Public and Private Key created for the DHT Key
DHT_KEY_CREDS = "syft-dht-key-creds"

USE_DIRECT_CONNECTION = True
MAX_MESSAGE_SIZE = 32768  # 32KB
MAX_STREAMER_CONCURRENCY = 200
TIMEOUT = 10  # in seconds
