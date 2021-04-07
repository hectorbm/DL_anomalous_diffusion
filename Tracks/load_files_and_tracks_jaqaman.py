from Tools.db_connection import connect_to_db, disconnect_to_db
from Tracks.file import get_files_in_path, File


def load_files_and_tracks(path_name):
    connect_to_db()
    print("Directory: ", end="")
    path = str(input())
    try:
        files = get_files_in_path(path_name=path_name)
        for f in files:

            try:
                file = File()
                print('Processing file:{}'.format(f))
                file.parse_filename(filename=f)
                file.add_raw_file(filename=f)
                file.save()
                tracks = file.load_local_file(filename=f)
                for track in tracks:
                    track.save()
            except AssertionError:
                print("Unable to load file:{}".format(f))
            finally:
                print("File loaded")
    except FileNotFoundError:
        print("Invalid directory")
    disconnect_to_db()