import os
import sys
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import datetime
from Model import *

# using Argument Parser to use different model generically
argparser = ArgumentParser()
argparser.add_argument("--model", default="LinearRegression", type=str)
argparser.add_argument("--pastDay", default=5, type=int)
args = argparser.parse_args()

model_constructor = eval(args.model)
model = model_constructor()
parameter_list = {}
if (args.model == "LinearRegression"):
    parameter_list["past_day"] = args.pastDay

# download data
os.system(
    "wget -N https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
os.system(
    "wget -N https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
os.system(
    "wget -N https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")

if (not os.path.exists("./prediction")):
    os.system("mkdir prediction")

confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
deaths = pd.read_csv('time_series_covid19_deaths_global.csv')
recovered = pd.read_csv('time_series_covid19_recovered_global.csv')

"""
# only study some countries
cp = pd.read_csv('country_to_study.csv')
countries = np.array(cp["Country"], dtype=str)
populations = np.array(cp["Populations"], dtype=float)
"""

col_names = confirmed.columns.tolist()
start = datetime.datetime.strptime(col_names[4], "%m/%d/%y")
end = datetime.datetime.strptime(col_names[-1], "%m/%d/%y")
date_in_the_past = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days + 1)]

for num_day_pred in [2, 7, 30]:
    last_date = date_in_the_past[-1]
    last_date_str = last_date.strftime("%Y-%m-%d")
    next_pred_date = last_date + datetime.timedelta(days=num_day_pred)
    file_str = 'prediction/' + str(num_day_pred) + "day_prediction_" + last_date.strftime("%Y-%m-%d") + ".csv"

    f = open(file_str, "w")
    f.write(
        "Province/State,Country,Target/Date,N,low95N,high95N,R,low95R,high95R,D,low95D,high95D,T,low95T,high95T,M,low95M,high95M,C,low95C,high95C\n")

    province_null_idx = confirmed['Province/State'].isnull()
    next_pred_date_str = next_pred_date.strftime("%Y-%m-%d") + ","

    for idx in range(0, confirmed.shape[0]):

        try:

            if (province_null_idx[idx] == True):
                loc1_str = ","
            else:
                loc1 = confirmed['Province/State'].iloc[idx]
                loc1_str = str(loc1).replace(',', ' ') + ","

            loc2 = confirmed['Country/Region'].iloc[idx]
            loc2_str = str(loc2).replace(',', ' ') + ","

            ###TODO: confident interval not yet implemented
            try:
                next_confirmed = model.predict(
                    confirmed.iloc[idx][col_names[4:]].to_numpy(dtype=float), num_day_pred,
                    parameter_list, loc2_str)
                n_str = str(next_confirmed) + ",,,"
            except:
                next_confirmed = float("NaN")
                n_str = ",,,"

            if (province_null_idx[idx] == True):
                country = str(confirmed.iloc[idx][col_names[1]])

                ###TODO: confident interval not yet implemented
                try:
                    # there is a bug that there are several states for some country that can cause problem
                    new_recovered = recovered.loc[recovered["Country/Region"] == country]
                    next_recovered = model.predict(
                        new_recovered.loc[new_recovered["Province/State"].isnull()][col_names[4:]].to_numpy(
                            dtype=float)[0], num_day_pred,
                        parameter_list, country)
                    next_recovered = min(next_confirmed, next_recovered)
                    r_str = str(next_recovered) + ",,,"
                except:
                    next_recovered = float("NaN")
                    r_str = ",,,"

                ###TODO: confident interval not yet implemented
                try:
                    # there is a bug that there are several states for some country that can cause problem
                    new_deaths = deaths.loc[deaths["Country/Region"] == country]
                    next_deaths = model.predict(
                        new_deaths.loc[new_deaths["Province/State"].isnull()][col_names[4:]].to_numpy(dtype=float)[
                            0], num_day_pred,
                        parameter_list, country)
                    next_deaths = min(next_deaths, next_confirmed)
                    try:
                        temp = int(new_deaths.loc[new_deaths["Province/State"].isnull()][col_names[-1]])
                        recover_temp = int(new_recovered.loc[new_recovered["Province/State"].isnull()][col_names[-1]])
                        next_recovered = min(next_recovered, 1.1 * (next_confirmed - temp))
                        next_deaths = min(next_deaths, next_confirmed - next_recovered)
                        if (next_deaths < temp):
                            next_deaths = round(1.1 * temp)
                            next_recovered = max(next_confirmed - next_deaths, recover_temp)
                            next_deaths = min(next_deaths, next_confirmed - next_recovered)
                        r_str = str(next_recovered) + ",,,"
                    except:
                        next_deaths = min(next_deaths, next_confirmed)
                    d_str = str(next_deaths) + ",,,"
                except:
                    next_deaths = float("NaN")
                    d_str = ",,,"

            else:
                province = str(confirmed.iloc[idx][col_names[0]])

                ###TODO: confident interval not yet implemented
                try:
                    next_recovered = model.predict(
                        recovered.loc[recovered["Province/State"] == province][col_names[4:]].to_numpy(dtype=float)[0],
                        num_day_pred, parameter_list, province)
                    next_recovered = min(next_confirmed, next_recovered)
                    r_str = str(next_recovered) + ",,,"
                except:
                    next_recovered = float("NaN")
                    r_str = ",,,"

                try:
                    next_deaths = model.predict(
                        deaths.loc[deaths["Province/State"] == province][col_names[4:]].to_numpy(dtype=float)[0],
                        num_day_pred, parameter_list, province)
                    next_deaths = min(next_deaths, next_confirmed)
                    try:
                        temp = int(deaths.loc[deaths["Province/State"] == province][col_names[-1]])
                        recover_temp = int(recovered.loc[recovered["Province/State"] == province][col_names[-1]])
                        next_recovered = min(next_recovered, int(1.1 * (next_confirmed - temp)))
                        next_deaths = min(next_deaths, next_confirmed - next_recovered)
                        if (next_deaths < temp):
                            next_deaths = round(1.1 * temp)
                            next_recovered = max(next_confirmed - next_deaths, recover_temp)
                            next_deaths = min(next_deaths, next_confirmed - next_recovered)
                            r_str = str(next_recovered) + ",,,"
                    except:
                        next_deaths = min(next_deaths, next_confirmed)
                    d_str = str(next_deaths) + ",,,"

                except:
                    next_deaths = float("NaN")
                    d_str = ",,,"

            # TODO: total number of tests not yet implemented
            t_str = ",,,"

            # TODO: mobility not yet implemented
            m_str = ",,,"

            # TODO: total number of critical cases not yet implemented
            c_str = ",,\n"

            try:
                f.write(loc1_str + loc2_str + next_pred_date_str + n_str + r_str + d_str + t_str + m_str + c_str)
            except:
                print("CAN NOT WRITE to file\n")

        except:
            print("Unexpected error:", sys.exc_info()[0])
    f.close()

print("finish predict for date: " + last_date.strftime("%Y-%m-%d"))
