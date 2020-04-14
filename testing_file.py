from tracks.file import File
from tools.db_connection import *
if __name__ == '__main__':
    filename = 'test_data/1-mAb35_CDx_0min.csv'
    file = File()
    connect_to_db()
    file.add_raw_file(filename=filename)
    file.save()
    tracks = file.load_local_file(filename=filename)
    for track in tracks:
        track.save()
    disconnect_to_db()
