import pmdarima as pm
from pmdarima.arima import auto_arima

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
                            start_P=0, seasonal=False,
                            d=1, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=False, 
                            stepwise=True)
    print(stepwise_model.aic())

def fit_summary(MODEL, train, order, freq):
    model=MODEL(train, order=order, freq=freq)
    model_fit=model.fit()
    print(model_fit.summary())
    return model_fit

def predict(model_fit, start, end, legende):
    pred=model_fit.predict(start=start,end=end,typ='levels').rename(legende)
    return pred