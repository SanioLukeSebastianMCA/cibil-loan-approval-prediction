import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from os.path import dirname, join


def loan_approval_predict(val1, val2, val3, val4, val5):
    c_file = join(dirname(__file__), "cibil_score_dataset.csv")
    cibil_data = pd.read_csv(c_file)
    X = cibil_data.iloc[:, :-1].values
    y = cibil_data.iloc[:, -1].values

    gnb = GaussianNB()
    gnb.fit(X, y.ravel())

    X_test = np.array([[val1, val2, val3, val4, val5]])
    y_pred = gnb.predict(X_test)
    return y_pred[0]


if __name__ == '__main__':
    print("The predicted value is : ",loan_approval_predict(1900, 1200, 1700, 2400, 0.8))
