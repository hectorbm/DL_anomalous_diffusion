import pandas as pd
import os
import pickle
def collapse_to_mean(data_df):

    group_data = dict()

    #Get all the values for each group
    print("Getting tracks (x,y) positions")
    for index, val in data_df.iterrows():

        x_val = val['x']
        y_val = val['y']
        track_id_val = str(val['track_id'])

        if track_id_val in group_data.keys():
            group_data[track_id_val]['x'].append(x_val)  
            group_data[track_id_val]['y'].append(y_val)
            group_data[track_id_val]['N'] = group_data[track_id_val]['N'] + 1
        else:
            group_data[track_id_val] = {'x':[x_val], 'y':[y_val],'N': 1}

    return group_data

def get_path_to_files():

    print("Please give relative path to files: ",end="")
    relative_path = input()
    absolute_path_dir = os.getcwd()+ "/" + relative_path

    res = os.listdir(absolute_path_dir)

    print("You choose this files: ", end="")
    print(res)

    return [absolute_path_dir,relative_path]

def run_over_batch(directory,save_header):
    files = os.listdir(directory[0])

    for filename in files:

        print("Reading: " + filename)
        filename_with_path = directory[1] + "/" + filename

        data_df = pd.read_csv(filename_with_path, sep=",", header=0)

        data = collapse_to_mean(data_df)

        # save to csv file
        print("Saving data to .pkl")

        filename_to_save = directory[1] + "/" + filename + '_tracks' + '.pkl'
        file_to_save = open(filename_to_save, 'wb')
        pickle.dump(data,file_to_save)
        file_to_save.close()

def main():
    """"
    Set batch to true if you want to proccess a directory
    Set save_header to true if you want the first row to have columns name
    """
    batch = False
    save_header = True

    if batch:
        path_to_files = get_path_to_files()
        run_over_batch(path_to_files,save_header)

    if not batch:
        print("Please input filename: ",end="")
        filename = input()

        print("Reading: " + filename)
        data_df = pd.read_csv(filename, sep=",", header=0)
        

        #collapse by mean position
        data = collapse_to_mean(data_df)

        #save to csv file
        print("Saving data to .pkl")

        filename_to_save = filename + '_tracks' + '.pkl'
        file_to_save = open(filename_to_save, 'wb')
        pickle.dump(data,file_to_save)
        file_to_save.close()


if __name__ == '__main__':
    main()