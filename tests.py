import os


def load_directory(path_name):
    with os.scandir(path_name) as it:
        print(it)
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                file_extension = entry.name.split('.')
                if file_extension[len(file_extension) - 1] == "csv":
                    print(''.join([path_name, entry.name]))


if __name__ == '__main__':
    load_directory(path_name="experimental_data/btx")
