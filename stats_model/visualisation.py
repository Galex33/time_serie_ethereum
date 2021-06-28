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

def if_stationarity(timeseries):
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
    result = adfuller(timeseries['Open'])
    print('Statistiques ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

def log_plot(df):
    df = np.log(df)
    plt.plot(df)
    return df, plt.plot(df)

def predict_ETH_confidence(start_date, data):
    pred = res_s.get_prediction(start=start_date, 
                          end='2021-07-04',
                          dynamic=True, full_results=True)

    pred_ci = pred.conf_int()

    plt.figure(figsize=(18, 8))

    # plot in-sample-prediction
    ax = data['2016-01-31':].plot(label='Real values',color='#006699');
    pred.predicted_mean.plot(ax=ax, label='ETH Predictions', alpha=.7, color='#ff0066');

    # draw confidence bound (gray)
    ax.fill_between(pred_ci.index, 
                    pred_ci.iloc[:, 0], 
                    pred_ci.iloc[:, 1], color='#ff0066', alpha=.25);

    # style the plot
    ax.fill_betweenx(ax.get_ylim(), start_date, data.index[-1], alpha=.15, zorder=-1, color='grey');
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.legend(loc='upper left')
    return plt.show()

def predict_future_price_ETH(model_fit, test_data):
    plt.figure(figsize=(12,8))
    start_date = input('Date de début de prédiction qui ne dépasse pas le 28 aout 2019 :')
    end_date = input('Date de fin de prédiction :')
    pred = model_fit.predict(start=start_date,end=end_date,typ='levels').rename('ARIMA Predictions')
    np.exp(pred).plot(legend=True, color='red')
    np.exp(test_data['Opening_Price_ETH']).plot(legend=True, label="Prix réel de l'ETH", color='green', style='--')
    plt.xlim(start_date, end_date)
    plt.ylim(0, np.exp(test_data['Opening_Price_ETH'][end_date]))
    plt.title("Prediction du prix de l'ETH en fonction de son cours")
    plt.xlabel("Date")
    plt.ylabel("Prix de l'ETH")
    return plt.show()
def predict_tomorrow_ETH_price(model_fit, df_log):
    plt.figure(figsize=(12,8))
    pred=model_fit.predict(start='2020-06-15',end='2021-08-15',typ='levels').rename('ARIMA Predictions')
    np.exp(pred).plot(legend=True, color='red')
    np.exp(df_log['Open'][len(df_log)-len(pred):]).plot(legend=True, label='Real values', color='blue', style='--')
    plt.ylabel("Cours de l'ETH(USD)")
    plt.title("Prédiction du cours de l'ETH(USD)")
    plt.show()

def predict_date(start_date, end_date, model_fit):
    print(f"Les prédictions du cours de l'ETH pour le {end_date} est de {np.exp(model_fit.predict(start=start_date,end=end_date,typ='levels')).values[-1]} dollars")