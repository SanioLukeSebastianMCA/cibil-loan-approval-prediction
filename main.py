import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from os.path import dirname, join


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    loan_approval_predict(1000,1500,2000,2400,0.4)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


def loan_approval_predict(val1, val2, val3, val4, val5):
    c_file= join(dirname(__file__),"cibil_score_dataset.csv")
    cibil_data = pd.read_csv(c_file)
    X = cibil_data.iloc[:, :-1].values
    y = cibil_data.iloc[:, -1].values

    gnb = GaussianNB()
    gnb.fit(X, y.ravel())

    X_test= np.array([[val1,val2,val3,val4, val5]])
    y_pred = gnb.predict(X_test)
    return y_pred[0]

# print("The approval value is : ",loan_approval_predict(1700,1900,1400,1600,0.83))

