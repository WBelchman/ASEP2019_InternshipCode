from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import numpy as np

import process_data
import metrics

class svm_model():
    def __init__(self):
        self.model = SVC(C=1.0, kernel='rbf', gamma=0.0001)

    def get_accuracy(self, labels, pred):
        f_positives = 0
        f_negatives = 0
        
        for t, p in zip(labels, pred):
            if p != t:
                if p > t:
                    f_positives += 1
                else:
                    f_negatives += 1

        return f_positives, f_negatives

    def fit(self, X_train, Y_train, X_test, Y_test):
        print("[*] Training model")

        self.model = self.model.fit(X_train, np.ravel(Y_train))

        print("[*] Evaluating model")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        print("\nTraining complete:")

        print('\nTraining set - predictions')
        print(np.ravel(Y_train))
        print(train_pred)

        print('\nTesting set - predictions')
        print(np.ravel(Y_test))
        print(test_pred)


        fp, fn = self.get_accuracy(Y_train, train_pred)
        train_acc = 1.0 - ((fp + fn) / len(train_pred))

        fp2, fn2 = self.get_accuracy(Y_test, test_pred)
        test_acc = 1.0 - ((fp2 + fn2) / len(test_pred))

        acc = 1.0 - ((fp + fn + fp2 + fn2) / (len(train_pred) + len(test_pred)))
        
        print("Training set accuracy - {}  F_p: {}  F_n: {}".format(round(train_acc, 3), fp, fn))
        print("Test set accuracy - {}  F_p: {}  F_n: {}".format(round(test_acc, 3), fp2, fn2))
        print("Accuracy - {}  F_p: {}  F_n: {}".format(round(acc, 3), (fp + fp2), (fn + fn2)))
        
        input("Press Enter to continue")

    def predict(self, x, y):
        return      

    def save(self):
        raise NotImplementedError

    def load(self, filename='default'):
        raise NotImplementedError

#Converts data into classes, 0 no crack, 0.9 crack has formed
def categorize(data, zero=False):
    for i, e in enumerate(data):
        if e > 0:
            data[i] = 1
        else:
            if zero:
                data[i] = 0
            else:
                data[i] = -1
    
    return data.astype('int')

#Handles training the model, allows for changing inputs easily
def train(model):
    data, _ = process_data.get_signal_values()
    Y = process_data.get_crack_lengths()

    X = metrics.fft_amp_sums(data)
    #X = metrics.correlation_coef(data)

    i = metrics.avg_peak_width(data)
    X = metrics.concatenate_data(X, i)

    i = metrics.correlation_coef(data) 
    X = metrics.concatenate_data(X, i)

    i = metrics.fft_amp_max(data) 
    X = metrics.concatenate_data(X, i)

    # i = metrics.xc_sub_signals(data)
    # X = metrics.concatenate_data(X, i)

    i = metrics.psd_height_sum(data)
    X = metrics.concatenate_data(X, i)

    i = metrics.xc_mean_bin1(data)
    X = metrics.concatenate_data(X, i)

    X_train, Y_train , X_test, Y_test = process_data.remove_one_plate(X, Y)
    X_train, Y_train , X_test, Y_test = process_data.flatten_data(X_train, Y_train , X_test, Y_test)

    scaler = RobustScaler().fit(np.concatenate((X_train, X_test)))

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    Y_train = categorize(Y_train, zero=True)
    Y_test = categorize(Y_test, zero=True)

    model.fit(X_train, Y_train, X_test, Y_test)

def predict(model):
    data, _ = process_data.get_signal_values()
    Y = process_data.get_crack_lengths()
    
    x1 = metrics.fft_amp_sums(data)
    x2 = metrics.avg_peak_width(data)

    X = metrics.concatenate_data(x1, x2)

    X, Y, _, _ = process_data.flatten_data(X, Y)

    predictions = model.predict(X, Y)

    print("\nTrue | Predicted")
    for t, p in zip(Y, predictions):
        print(t, "-", p)


#Mainly just UI
def run():
    model = svm_model()

    print("\nPHM Crack Detection SVM")
    
    while True:
        user = input("1. Train\n2. Predict\n3. Save\n4. Load\n5. Quit\n")
        
        if user == '1':
            train(model)
        elif user == '2':
            predict(model)
        elif user == '3':
            model.save()
        elif user == '4':
            model.load()
        elif user == '5':
            break

        print("\n-----------SVM------------")