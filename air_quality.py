import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('AirQualityUCI.csv', decimal=',', sep=';')

df.replace(-200, np.nan, inplace=True)
df.dropna(subset = ['CO(GT)','PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH'], inplace=True)
df.dropna(how='all', inplace=True)

x_train,x_test,y_train,y_test = train_test_split(
    df[['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']],
    df['CO(GT)'],
    test_size=0.2,
    random_state=42
)

train_mean = x_train.mean()
train_std = x_train.std()
x_train = (x_train - train_mean)/train_std
x_test = (x_test - train_mean)/train_std

def train():
    n = len(x_train)
    m = np.zeros(x_train.shape[1])
    b = 0
    lr = 0.01
    training = 10000

    for _ in range(training):
        X = x_train.values
        y_train_pred = np.dot(X, m)+b

        dm = (-2/n)*np.dot(X.T, (y_train-y_train_pred))
        db = (-2/n)*np.sum(y_train-y_train_pred)

        m = m - lr*dm
        b = b - lr*db

    y_train_pred = np.dot(X,m)+b
    train_mse = np.mean((y_train-y_train_pred)**2)
    return m,b,train_mse,y_train_pred

m,b,train_mse,y_pred = train()
y_test_pred = np.dot(x_test.values,m)+b
test_mse = np.mean((y_test-y_test_pred)**2)
r2 = 1 - (np.sum((y_test-y_test_pred)**2))/np.sum((y_test-np.mean(y_test))**2)

print('R2-score: ',r2)
print('Train MSE: ',train_mse)
print('Test MSE: ',test_mse)

# new data for prediction
new_data = np.array(list(map(float,input("Enter new data: ").split()))).reshape(1,-1)
new_data_scaled = (new_data - train_mean.values) / train_std.values
pred = np.dot(new_data_scaled,m)+b
print(pred)