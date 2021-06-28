import pmdarima as pm
from pmdarima.arima import auto_arima
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import special

def train_test(df, size_float):
    X = df
    y = df.values
    size = int(len(X) * size_float)
    train, test = X[0:size], X[size:len(X)]
    print(f'Train : {train.shape}, Test : {test.shape}')
    return y, train, test

def auto_arima(train):
    stepwise_model = pm.auto_arima(train, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=False, 
                            stepwise=True)
    print(stepwise_model.aic())

def fit_summary(df):
    model=sm.tsa.SARIMAX(df, order=(1,1,0),seasonal_order=(2,1,0,12), enforce_stationarity=True, enforce_invertibility=True)
    model_fit=model.fit()
    print(model_fit.summary())
    return model_fit

def predict_ETH_confidence(start_date, data, model_fit):
    pred = model_fit.get_prediction(start=start_date, 
                          end='2021-07-04',
                          dynamic=True, full_results=True)

    pred_ci = pred.conf_int()

    plt.figure(figsize=(18, 8))

    # plot in-sample-prediction
    ax = data['2016-01-31':].plot(label='Observed',color='#006699');
    pred.predicted_mean.plot(ax=ax, label='One-step Ahead Prediction', alpha=.7, color='#ff0066');

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

def pred_ci(model_fit, start_date, end_date):
    pred = model_fit.get_prediction(start=start_date, end=end_date)
    pred_ci = pred.conf_int()
    return pred, pred_ci

def inverse_box_cox(model_fit):
    pred = model_fit.get_prediction(start='2021-06-27', end='2021-07-04')
    pred_ci_orig = special.inv_boxcox(pred.conf_int(), 0)
    forecast = special.inv_boxcox(pred.predicted_mean, 0)
    return forecast
    
def display_predictions(df, model_fit):
    pred = model_fit.get_prediction(start='2021-06-27', end='2021-12-20')
    pred_ci_orig = special.inv_boxcox(pred.conf_int(), 0)
    forecast = special.inv_boxcox(pred.predicted_mean, 0)
    ax = df.plot(label='observed')
    ax.figure.set_size_inches(12, 8)
    forecast.plot(ax=ax, label='forecast', lw=3, alpha=.7, color='#ff0066')
    ax.fill_between(pred_ci_orig.index,
                    pred_ci_orig.iloc[:, 0],
                    pred_ci_orig.iloc[:, 1], color='k', alpha=.15)
    ax.set_title('Forecast in original units (Mean)', fontsize=16)
    plt.ylim(0,5000)
    plt.legend()
    return plt.show();

def predict(model_fit, start, end, legende):
    pred=model_fit.predict(start=start,end=end,typ='levels').rename(legende)
    return pred

def predict_special_date(model_fit, start_date, end_date):
    prediction_values = model_fit.get_prediction(start=start_date, end=end_date)
    pred_ci = prediction_values.conf_int()
    forecast_values = special.inv_boxcox(prediction_values.predicted_mean, 0)
    print(f"Le cours de l'ETH du à partir du {start_date} sera de \n{forecast_values[0]} dollars")