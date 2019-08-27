import numpy as np
import pandas as pd
from pathlib import Path

def from_csv(filename):
    filename = 'PHM_Data_Challenge_19/PHM2019_crack_growth_training_data/' + filename
    return np.matrix(pd.read_csv(filename))

def from_excel(filename):
    filename = 'PHM_Data_Challenge_19/PHM2019_crack_growth_training_data/' + filename
    return np.matrix(pd.read_excel(filename))

def strip_crack_length_values(file_data):
    file_data = file_data[5:, 1]

    #filters nan values
    for i2 in range(len(file_data)):
        if pd.isnull(file_data[i2, 0]):
            data = file_data[:i2]
            break

    return data

#Elementwise mean of the two received signal, 
#there are no major feature differences between the two
def average_signals(df_1, df_2):
    mean_ch2 = (df_1[:, 2] + df_2[:, 2]) / 2.0
    df_1[:, 2] = mean_ch2       
    return df_1

def get_signal_values():

    data = []
    names = []

    for i in range(1, 8):
        plate_names = []
        plate_signals = []

        cycle_dirs = [str(f).replace("PHM_Data_Challenge_19/PHM2019_crack_growth_training_data/", '') 
                    for f in Path("PHM_Data_Challenge_19/PHM2019_crack_growth_training_data/T" + str(i)).iterdir()
                        if f.is_dir()]    

        cycle_dirs.sort(reverse=True)

        for path in cycle_dirs:                        

            d1 = from_csv(path + "/signal_1.csv")
            d2 = from_csv(path + "/signal_2.csv")
            d = average_signals(d1, d2)

            path = path.replace("/", '-')
            
            plate_names.append(path)
            plate_signals.append(d)

        names.append(plate_names)
        data.append(plate_signals)

    return data, names

def get_single_signal(filename):

    plate_names = []

    d1 = from_csv(filename + "/signal_1.csv")
    d2 = from_csv(filename + "/signal_2.csv")
    d = average_signals(d1, d2)

    filename = filename.replace("/", '-')
            
    plate_names.append(filename)

    return d, plate_names
    

def get_crack_lengths():
    data = [None] * 7
    
    for i in range(1,7):
        raw = from_excel("T" + str(i) + "/Description_T" + str(i) + ".xlsx")
        data[i-1] = strip_crack_length_values(raw)

    data[6] = [0] * 5

    return data

#Splits the data to remove a plate from the training set
#default chooses random plate to remove
def remove_one_plate(X, Y, plate=0):
    if plate == 0:
        plate = np.random.randint(0, 6)
    else:
        plate -= 1

    X_test = X[plate]
    Y_test = Y[plate]

    X_train = np.delete(X, plate)
    Y_train = np.delete(Y, plate)

    return X_train, Y_train, X_test, Y_test

#0 is just a placeholder value if X_test and Y_test are not needed
def flatten_data(X_train, Y_train, X_test=[], Y_test=[]):

    X_train = np.concatenate(X_train, axis=0)
    if X_test != []:
        X_test = np.array(X_test)

    Y_train = np.row_stack(Y_train).reshape(-1, 1)
    if Y_test != []:
        Y_test = np.row_stack(Y_test).reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test
