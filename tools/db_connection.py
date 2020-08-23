from mongoengine import connect
from mongoengine import disconnect


def connect_to_db():
    connect(db='tracking', host='127.0.0.1', port=27017)


def disconnect_to_db():
    disconnect()
