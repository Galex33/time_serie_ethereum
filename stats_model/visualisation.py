import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pylab import rcParams
from arima import *

def plot(df, date_start, date_end, column, currency):
    plt.figure(figsize=(12, 8))
    df.loc[date_start  : date_end, column].plot()
    df.loc[date_start : date_end, column].resample('W').mean().plot(label='moyenne par semaine', lw=2, ls='--', alpha=0.8)
    df.loc[date_start : date_end, column].resample('M').mean().plot(label='moyenne par mois', lw=3, ls=':', alpha=0.8)
    plt.legend()
    plt.ylabel(currency, fontsize=18)
    plt.xlabel('YEAR-MONTH', fontsize=18)
    return plt.show()

def get_stationarity(timeseries):
    rcParams['figure.figsize'] = 11, 9
    rcParams['lines.linewidth'] = 0.5
    # Statistiques mobiles
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    # tracé statistiques mobiles
    original = plt.plot(timeseries, color='blue', label='Origine')
    mean = plt.plot(rolling_mean, color='red', label='Moyenne Mobile')
    std = plt.plot(rolling_std, color='black', label='Ecart-type Mobile')
    plt.legend(loc='best')
    plt.title('Moyenne et écart-type Mobiles')
    plt.show(block=False)
    
    # Test Dickey–Fuller :
    result = adfuller(timeseries['Opening_Price_ETH'])
    print('Statistiques ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

def log_plot(df):
    df = np.log(df)
    plt.plot(df)
    return df, plt.plot(df)

def vis_predict(pred, test, legende):
    np.exp(pred).plot(legend=True)
    np.exp(test[legende]).plot(legend=True)
    plt.show()

def predict_future_price_ETH(model_fit, test_data):
    plt.figure(figsize=(12,8))
    start_date = input('Date de début de prédiction qui ne dépasse pas le 28 aout 2019 :')
    end_date = input('Date de fin de prédiction :')
    pred = model_fit.predict(start=start_date,end=end_date,typ='levels').rename('ARIMA Predictions')
    np.exp(pred).plot(legend=True)
    np.exp(test_data['Opening_Price_ETH']).plot(legend=True, label="Prix réel de l'ETH", color='green')
    plt.xlim(start_date, end_date)
    plt.ylim(0, np.exp(test_data['Opening_Price_ETH'][end_date]))
    plt.title("Prediction du prix de l'ETH en fonction de son cours")
    plt.xlabel("Date")
    plt.ylabel("Prix de l'ETH")
    return plt.show()