from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

import process_data
import metrics


class rf_model():
    def __init__(self, num_estimators=10000):
        self.rf = RandomForestRegressor(num_estimators)

    def fit(self, X_train, Y_train, X_test, Y_test):
        print("[*] Training model")
        self.rf.fit(X_train, Y_train)

        print("[*] Evaluating")
        self.predict(X_train, Y_train, X_test, Y_test)

    def predict(self, X_train, Y_train, X_test, Y_test):
        train_pred = self.rf.predict(X_train)
        test_pred = self.rf.predict(X_test)

        print('\nTraining set - predictions')
        print(Y_train)
        print(np.round(train_pred, 2))

        print('\nTesting set - predictions')
        print(Y_test)
        print(np.round(test_pred, 2))

        train_rmse = np.sqrt(np.mean((train_pred - Y_train) ** 2))

        test_rmse = np.sqrt(np.mean((test_pred - Y_test) ** 2))

        rmse = np.sqrt(np.mean(( \
            np.concatenate((train_pred, test_pred), axis=0) - \
            np.concatenate((Y_train, Y_test), axis=0))** 2))

        print("Estimator evaluation:")
        print("Training set RMSE - {}".format(round(train_rmse, 3)))
        print("Test set RMSE - {}".format(round(test_rmse, 3)))
        print("RMSE - {}".format(round(rmse, 3)))
        input("Press Enter to continue")         

    def save(self, filename):
        print('[*] Saving model to {}'.format(filename))
        pickle.dump(self.rf, open(filename, 'wb'))
        print('[*] Model saved')

    def load(self, filename='default'):
        print('[*] Loading model from {}'.format(filename))
        self.rf = pickle.load(open(filename, 'rb'))
        print('[*] Model loaded')

def load_data():
    data, _ = process_data.get_signal_values()
    Y = process_data.get_crack_lengths()

    X = metrics.fft_amp_sums(data)

    # i = metrics.avg_peak_width(data)
    # X = metrics.concatenate_data(X, i)

    i = metrics.correlation_coef(data) 
    X = metrics.concatenate_data(X, i)

    i = metrics.xc_mean_bin1(data) 
    X = metrics.concatenate_data(X, i)

    return X, Y

#Handles training the model, allows for changing metrics 'easily'
def train(model):    
    X, Y = load_data()

    X_train, Y_train , X_test, Y_test = process_data.remove_one_plate(X, Y, plate=7)
    X_train, Y_train , X_test, Y_test = process_data.flatten_data(X_train, Y_train , X_test, Y_test)

    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)

    print(X_test)

    model.fit(X_train, Y_train, X_test, Y_test)

def predict(model):
    X, Y = load_data()

    X_train, Y_train , X_test, Y_test = process_data.remove_one_plate(X, Y)
    X_train, Y_train , X_test, Y_test = process_data.flatten_data(X_train, Y_train , X_test, Y_test)

    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)

    model.predict(X_train, Y_train, X_test, Y_test)

def save(model):
    while True:
        user = input("Would you like to specify a filename? y/n\n")

        if user == 'y':
            filename = input("Filename:\n")
            break
        elif user == 'n':
            filename = 'default'
            break

    model.save('RF_models/' + filename + '.sav')

def load(model):
    filename = input('Please enter filename\n')
    model.load('RF_models/' + filename + '.sav')

#Mainly just UI
def run():
    model = rf_model()

    print("\nPHM Random Forest Regression")
    
    while True:
        user = input("1. Train\n2. Predict\n3. Save\n4. Load\n5. Quit\n")
        
        if user == '1':
            train(model)
        elif user == '2':
            predict(model)
        elif user == '3':
            save(model)
        elif user == '4':
            load(model)
        elif user == '5':
            break

        print("\n-----------RF------------")