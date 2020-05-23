import numpy as np


class LinearRegression:
    """
    A naive model for forecasting.
    It is believed that the 30 days prediction is definitely unreliable 
    """
    def __init__(self):
        pass

    def predict(self, past_data, date_to_predict, parameter_list, place=""):
        assert date_to_predict in [2, 7, 30]
        if(date_to_predict == 2):
            return self.predict_2day(past_data, parameter_list, place)
        elif(date_to_predict == 7):
            return self.predict_7day(past_data, parameter_list)
        else:
            return self.predict_30day(past_data, parameter_list)

    def predict_2day(self, past_data, parameter_list, place=""):
        past_day = parameter_list["past_day"]
        x = np.arange(past_day)
        y = past_data[-past_day:]
        ## Check if there is bug in the given data that have a reduce in total confirmed case
        for i in range(len(y)-1):
            if(y[i] > y[i+1]):
                print("Bug in the input data. Please check !!!!!!!!!!!!!!!!!")
                print("check " + place)
        fit = np.polyfit(x, y, 1)
        f = np.poly1d(fit)
        # a direct fit to after 2 days
        return int(f(past_day+1)+0.5)

    def predict_7day(self, past_data, parameter_list):
        past_day = parameter_list["past_day"]
        x = np.arange(past_day)
        y = past_data[-past_day:]
        # a recursive linear fit after single day fitting
        for i in range(7):
            fit = np.polyfit(x, y, 1)
            f = np.poly1d(fit)
            y[:-1] = y[1:]
            y[-1] = f(past_day)
        return int(y[-1]+0.5)

    # exactly same as the function for 7 days
    def predict_30day(self, past_data, parameter_list):
        past_day = parameter_list["past_day"]
        x = np.arange(past_day)
        y = past_data[-past_day:]
        # a recursive linear fit after single day fitting
        for i in range(30):
            fit = np.polyfit(x, y, 1)
            f = np.poly1d(fit)
            y[:-1] = y[1:]
            y[-1] = f(past_day)
        return int(y[-1]+0.5)

