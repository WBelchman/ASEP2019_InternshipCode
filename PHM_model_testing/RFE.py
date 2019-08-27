from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import numpy as np

import process_data
import metrics

class rfe_model():
    def __init__(self):
        self.svr = SVR(C=1.0, kernel='linear', gamma='scale')
        self.model = RFE(self.svr, n_features_to_select=3, step=1)

    def fit(self, X_train, Y_train, X_test, Y_test):
        print("[*] Training model")

        self.model = self.model.fit(X_train, Y_train)

        print("[*] Evaluating model")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        print("\n[*] Training complete:")

        print('Feature selection:')
        print(self.model.ranking_)

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
        print("Training set rmse - {}".format(round(train_rmse, 3)))
        print("Test set rmse - {}".format(round(test_rmse, 3)))
        print("RMSE - {}".format(round(rmse, 3)))
        
        input("Press Enter to continue")

    def train_SVM(self, X_train, Y_train, X_test, Y_test):

        print("[*] Training model")

        self.svr = self.svr.fit(X_train, Y_train)

        print("[*] Evaluating model")
        train_pred = self.svr.predict(X_train)
        test_pred = self.svr.predict(X_test)

        print("\n[*] Training complete:")

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
        print("Training set rmse - {}".format(round(train_rmse, 3)))
        print("Test set rmse - {}".format(round(test_rmse, 3)))
        print("RMSE - {}".format(round(rmse, 3)))
        
        input("Press Enter to continue")    

    def save(self):
        raise NotImplementedError

    def load(self, filename='default'):
        raise NotImplementedError

#Handles training the model, allows for changing inputs easily
def train(model):
    data, _ = process_data.get_signal_values()
    Y = process_data.get_crack_lengths()

    X = metrics.fft_amp_sums(data) #yes
    #X = metrics.correlation_coef(data)

    i = metrics.avg_peak_width(data) #no
    X = metrics.concatenate_data(X, i)

    i = metrics.correlation_coef(data) #yes
    X = metrics.concatenate_data(X, i)

    i = metrics.fft_amp_max(data) #no
    X = metrics.concatenate_data(X, i)

    i = metrics.psd_height_sum(data) #maybe
    X = metrics.concatenate_data(X, i)

    i = metrics.xc_mean_bin1(data) #yes
    X = metrics.concatenate_data(X, i)

    # i = metrics.psd_max_height(data) 
    # X = metrics.concatenate_data(X, i)

    X_train, Y_train , X_test, Y_test = process_data.remove_one_plate(X, Y)
    X_train, Y_train , X_test, Y_test = process_data.flatten_data(X_train, Y_train , X_test, Y_test)

    scaler = RobustScaler().fit(np.concatenate((X_train, X_test)))

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)

    model.fit(X_train, Y_train, X_test, Y_test)

def run_SVM(model):
    data, _ = process_data.get_signal_values()
    Y = process_data.get_crack_lengths()

    X = metrics.fft_amp_sums(data) #yes
    #X = metrics.correlation_coef(data)

    i = metrics.correlation_coef(data) #yes
    X = metrics.concatenate_data(X, i)

    i = metrics.psd_height_sum(data) #maybe
    X = metrics.concatenate_data(X, i)

    i = metrics.xc_mean_bin1(data) #yes
    X = metrics.concatenate_data(X, i)

    X_train, Y_train , X_test, Y_test = process_data.remove_one_plate(X, Y)
    X_train, Y_train , X_test, Y_test = process_data.flatten_data(X_train, Y_train , X_test, Y_test)

    scaler = RobustScaler().fit(np.concatenate((X_train, X_test)))

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)

    model.train_SVM(X_train, Y_train, X_test, Y_test)


#Mainly just UI
def run():
    model = rfe_model()

    print("\nPHM Recursive Feature Elimination")
    
    while True:
        user = input("1. Train\n2. Run SVM\n3. Save\n4. Load\n5. Quit\n")
        
        if user == '1':
            train(model)
        elif user == '2':
            run_SVM(model)
        elif user == '3':
            model.save()
        elif user == '4':
            model.load()
        elif user == '5':
            break

        print("\n-----------RFE------------")