from keras.models import load_model


def load_model_from_file(filename):
    try:
        model = load_model(filename, compile=True)
    except ValueError:
        print("File does not exist!")
    return model
