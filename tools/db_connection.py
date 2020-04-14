from mongoengine import connect
from mongoengine import disconnect


def connect_to_db():
    connect(db='tracking', host='172.17.0.2', port=27017)


def disconnect_to_db():
    disconnect()
