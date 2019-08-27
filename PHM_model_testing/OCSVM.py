from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
import numpy as np

import process_data
import metrics

def test(X, Y):
    rem = [] 
    for i, y in enumerate(Y):
        if y == -1:
            rem.append(i)
                      
    X = np.delete(X, rem, axis=0)
    return X

class ocsvm_model():
    def __init__(self):
        self.model = OneClassSVM(kernel='rbf', gamma=0.1)

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
        self.model = self.model.fit(test(X_train, Y_train))

        print("[*] Evaluating model")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

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

    def predict(self, x, y):
        return      

    def save(self):
        raise NotImplementedError

    def load(self, filename='default'):
        raise NotImplementedError

#Converts data into classes, 0 no crack, 0.9 crack has formed
def categorize(data):
    for i, e in enumerate(data):
        if e > 0:
            data[i] = 1
        else:
            data[i] = -1
    
    return data

#Handles training the model, allows for changing inputs easily
def train(model):
    data, _ = process_data.get_signal_values()
    Y = process_data.get_crack_lengths()

    X = metrics.fft_amp_sums(data) #yes
    #X = metrics.correlation_coef(data)

    # i = metrics.avg_peak_width(data) #no
    # X = metrics.concatenate_data(X, i)

    i = metrics.correlation_coef(data) #yes
    X = metrics.concatenate_data(X, i)

    # i = metrics.fft_amp_max(data) #no
    # X = metrics.concatenate_data(X, i)

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

    Y_train = categorize(Y_train)
    Y_test = categorize(Y_test)

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
    model = ocsvm_model()

    print("\nPHM Crack Detection OCSVM")
    
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

        print("\n----------OCSVM-----------")