from mongoengine import connect
from mongoengine import disconnect

db_name = 'tracking'
db_ip = '127.0.0.1'
db_port = 27017


def connect_to_db():
    connect(db=db_name, host=db_ip, port=db_port)


def disconnect_to_db():
    disconnect()
