# import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation  # , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df = None


# selects the style
def select_style(style_name):
    style.use(style_name)


# gets the stock data for a given stock name
def get_stock_db(stock_name):
    global df
    df = quandl.get('WIKI/' + stock_name)

# get the stock data
get_stock_db('GOOGL')

# narrows down to necessary features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# creates columns using calculations from other columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# these are the features
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

# filling NaN values with a large inexplicable number
# to show outlier values
# --> necessary for algorithm I think?
df.fillna(-9999999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

# shifts the days by forecast_out days
df['Next Day'] = df[forecast_col].shift(-forecast_out)

# creating an array that doesn't have next day column values
x = np.array(df.drop(['Next Day'], 1))
x = preprocessing.scale(x)

# forecast_out days before x --> used for training
x = x[:-forecast_out]

# forecast_out days after x --> used for predicting
x_after = x[-forecast_out:]

# drop values that don't have a value
df.dropna(inplace=True)

# y is the output of the function --> next day values
y = np.array(df['Next Day'])

# cross_validation.train_test_split results in two arrays
#       1. An array of x-y values used for training
#       2. An array of x-y values used for testing
# this is to avoid testing and training with the same
# arrays all the time --> it'll fit only one curve
# while you want it to be able to work with a variety
# of data
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

# this is the classifier used --> linear regression
clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)

with open('linearreg.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_input = open('linearreg.pickle', 'rb')
clf = pickle.load(pickle_input)

# fitting the trained data to a line found by linear regression

# comparing the data predicted with the test values with the actual values
accuracy = clf.score(x_test, y_test)

# predicting future values
forecast_set = clf.predict(x_after)

print(forecast_set, accuracy, forecast_out)

# Setting forecast values to nan
df['Forecast'] = np.nan

# hard-coding date into x-axis for the graph
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# setting style and details of the graph
select_style('seaborn-pastel')
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
