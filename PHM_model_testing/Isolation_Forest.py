from sklearn.ensemble import IsolationForest
import numpy as np
import pickle

import process_data
import metrics


class isf_model():
    def __init__(self, num_estimators=10000):
        self.isf = IsolationForest(num_estimators, contamination='auto', behaviour='new')

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
        self.isf.fit(X_train)

        print("[*] Evaluating")
        self.predict(X_train, Y_train, X_test, Y_test)

    def predict(self, X_train, Y_train, X_test, Y_test):
        train_pred = self.isf.predict(X_train)
        test_pred = self.isf.predict(X_test)

        fp, fn = self.get_accuracy(Y_train, train_pred)
        train_acc = 1.0 - ((fp + fn) / len(train_pred))

        fp2, fn2 = self.get_accuracy(Y_test, test_pred)
        test_acc = 1.0 - ((fp2 + fn2) / len(test_pred))

        acc = 1.0 - ((fp + fn + fp2 + fn2) / (len(train_pred) + len(test_pred)))

        print("\nTraining complete:")
        print("Training set accuracy - {}  F_p: {}  F_n: {}".format(round(train_acc, 3), fp, fn))
        print("Test set accuracy - {}  F_p: {}  F_n: {}".format(round(test_acc, 3), fp2, fn2))
        print("Accuracy - {}  F_p: {}  F_n: {}".format(round(acc, 3), (fp + fp2), (fn + fn2)))
        
        input("Press Enter to continue")        

    def save(self, filename):
        print('[*] Saving model to {}'.format(filename))
        pickle.dump(self.rf, open(filename, 'wb'))
        print('[*] Model saved')

    def load(self, filename='default'):
        print('[*] Loading model from {}'.format(filename))
        self.isf = pickle.load(open(filename, 'rb'))
        print('[*] Model loaded')

def categorize(data):
    for i, e in enumerate(data):
        if e > 0:
            data[i] = 1
        else:
            data[i] = -1
    
    return data

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

    X_train, Y_train , X_test, Y_test = process_data.remove_one_plate(X, Y)
    X_train, Y_train , X_test, Y_test = process_data.flatten_data(X_train, Y_train , X_test, Y_test)

    Y_train = categorize(Y_train)
    Y_test = categorize(Y_test)

    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)

    model.fit(X_train, Y_train, X_test, Y_test)

def predict(model):
    X, Y = load_data()

    X_train, Y_train , X_test, Y_test = process_data.remove_one_plate(X, Y)
    X_train, Y_train , X_test, Y_test = process_data.flatten_data(X_train, Y_train , X_test, Y_test)

    Y_train = categorize(Y_train)
    Y_test = categorize(Y_test)

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
    model = isf_model()

    print("\nPHM Random Forest Classifier (Outlier detection)")
    
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