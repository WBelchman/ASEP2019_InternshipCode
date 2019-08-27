from sklearn.preprocessing import RobustScaler
import autograd
import autograd.numpy as np2
import numpy as np
from math import e

import process_data
import metrics


class lin_reg():

    def fit(X_train, Y_train, X_test, Y_test):

        full_crack_length = 7.0

        eps = 1e-15
        weights = np.zeros(X_train.shape[1])

        def wTx(w, x):
            return np2.dot(x, w)

        def sigmoid(z):
            return 1./(1+(e**(-z)))

        def logistic_predictions(w, x):
            predictions = sigmoid(wTx(w, x))
            return predictions.clip(eps, 1-eps)

        def custom_loss(y, pred_y):
            A = []
            M = []
            prev_pred_y = 0
            AB = False

            T = np.squeeze(2.0 + 10.0 * (y/full_crack_length))

            if type(pred_y) != np.ndarray:
                node = pred_y._node
                trace = pred_y._trace
                AB = True

            pred_y = [np.round(_) for _ in pred_y]   
            
            for t, p in zip(y, pred_y):

                t = t/full_crack_length
                p = p/full_crack_length
                
                if p - t >= 0:
                    A.append(e**(np.abs(p - t)/0.5))
                elif p - t < 0:
                    A.append(e**(np.abs(p - t)/0.2))

                if p - prev_pred_y < 0:
                    M.append(1.0 + 10.0 * (np.abs(p - prev_pred_y)))
                elif p - prev_pred_y >= 0:
                    M.append(1.0)

                prev_pred_y = p

            J = np.mean(T) * np.mean(A) * np.mean(M)

            if AB:
                return np2.numpy_boxes.ArrayBox(J, trace, node)
            else:
                return J

        def custom_loss_with_weights(w):
            y_predicted = logistic_predictions(w, X_train)
            return custom_loss(Y_train, y_predicted)


        gradient = autograd.grad(custom_loss_with_weights)

        print("[*] Training model")

        for i in range(50000):
            weights -= gradient(weights) * 0.00001
            if i % 1000 == 0:
                print('Iterations {} | Loss {}'.format(i, custom_loss_with_weights(weights)))
                print(weights)

        print("[*] Evaluating model")
        train_pred = wTx(weights, X_train)
        test_pred = wTx(weights, X_test)

        print("\n[*] Training complete:")

        print('\nTraining set - predictions')
        print(np.squeeze(Y_train))
        print(np.round(train_pred, 2))

        print('\nTesting set - predictions')
        print(np.squeeze(Y_test))
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

    def predict(self, x, y):
        raise NotImplementedError      

    def save(self):
        raise NotImplementedError

    def load(self, filename='default'):
        raise NotImplementedError

#Handles training the model, allows for changing inputs easily
def train():
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

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    lin_reg.fit(X_train, Y_train, X_test, Y_test)

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

    print("\nPHM Linear Regression")
    
    while True:
        user = input("1. Train\n2. Predict\n3. Save\n4. Load\n5. Quit\n")
        
        if user == '1':
            train()
        elif user == '2':
            predict()
        elif user == '3':
            lin_reg.save()
        elif user == '4':
            lin_reg.load()
        elif user == '5':
            break

        print("\n--------Linear Regression---------")