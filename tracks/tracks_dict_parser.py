def create_tracks_dict(data):
    data_dict = dict()
    for index, row in data.iterrows():
        particle_id = row['track_id']
        x = float(row['x'])
        y = float(row['y'])
        frame = int(row['frame'])
        if particle_id in data_dict:
            data_dict[particle_id]["x"].append(x)
            data_dict[particle_id]["y"].append(y)
            data_dict[particle_id]['frame'].append(frame)
        else:
            data_dict[particle_id] = {"x": [x], "y": [y], 'frame': [frame]}

    data_dict.pop(0)  # Remove non assigned particles

    return data_dict
