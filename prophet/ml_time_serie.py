from prophet import Prophet
from datetime import date
from datetime import timedelta 

def model_fit(seasonality_mode, df):
    model = Prophet(seasonality_mode=seasonality_mode)
    model = model.fit(df);
    return model

def model_predict_period(model, periods):
    prediction = model.make_future_dataframe(periods = periods)
    return prediction

def model_predict(model,prediction):
    forecast = model.predict(prediction)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
    return forecast

def predict(forecast):
    next_day = (date(2021, 1, 1)) + timedelta(days=1)
    forecast[forecast['ds'] == next_day]['yhat']


def time_serie(seasonality_mode, df, periods):
    model = model_fit(seasonality_mode, df)
    prediction = model_predict_period(model, periods)
    forecast = model_predict(model,prediction)
    predict(forecast)
    return model, forecast