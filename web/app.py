from __future__ import division, print_function
import itertools
from flask import Flask, render_template
import sys
import os
import glob
import re
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import statsmodels.api as sm
import pickle

# tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
API_KEY='e84d5c69-af7c-49e3-b604-ad534ca5c7fe'

API_ENDPOINT = "https://api.airvisual.com/v2/city"

model = tf.keras.models.load_model('model/waste.h5')
pred_mod = pickle.load(open('model.pkl','rb'))

def display_stats():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('cleaned_dataset.csv')
    # calculate the mean AQI for each city
    mean_aqi_by_city = df.groupby('City')['AQI'].mean()

    # create a bar graph
    plt.figure(figsize=(10, 7))
    plt.bar(mean_aqi_by_city.index, mean_aqi_by_city.values)

    # set the title and axis labels
    plt.title('Mean AQI by City')
    plt.xlabel('City')
    plt.ylabel('Mean AQI')

    # show the plot
    plt.plot()
    if os.path.exists('static/plot.png'):
        os.remove('static/plot.png')
    plt.savefig('static/plot.png')

def district_stats(district):
    df = pd.read_csv('cleaned_dataset.csv')
    df_city = df[df.iloc[:, 0] == '{0}'.format(district)]

    # calculate the mean of each column
    mean_values = df_city[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']].mean()
    plt.figure(figsize=(10, 10))
    # create a pie chart
    plt.pie(mean_values.values, labels=mean_values.index, autopct='%1.1f%%')

    # add a legend
    plt.legend(mean_values.index, title='Pollutants', loc='best')

    # set the title
    plt.title('Mean Values of Air Pollutants')

    # show the plot
    plt.plot()

    if os.path.exists('static/district.png'):
        os.remove('static/district.png')
    plt.savefig('static/district.png')
    plt.close()

    # convert the 'Date' column to a datetime type
    df_city['Date'] = pd.to_datetime(df_city['Date'], format='%d-%m-%Y')

    # resample the data to 30-day intervals and calculate the mean AQI for each interval
    interval_data = df_city.resample('{0}D'.format(len(df_city)/10), on='Date')['AQI'].mean()

    plt.figure(figsize=(12, 8))

    # create a line graph
    plt.plot(interval_data.index, interval_data.values)

    # set the title and axis labels
    plt.title('Average Air Quality Index over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI')

    # show the plot
    plt.plot()
    if os.path.exists('static/date.png'):
        os.remove('static/date.png')
    plt.savefig('static/date.png')
    plt.close()
    
def model_predict(img_path,model):

    img = image.load_img(img_path,target_size=(200,235))
    #Preprocessing the image
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)

    result = model.predict(x)
    index = np.argmax(result)
    print(result)

    

    pred = ["E-Waste", "Food Waste", "Leaf-Waste", "Metal-Cans", "Paper-Waste", "Plastic-Bags","Plastic-Bottles","Wood-Waste" ]
    waste = pred[index]

   
    if waste == 'E-Waste':
       return redirect(url_for('ewaste'))
    if waste == 'Food-Waste':
       return redirect(url_for('food'))
    if waste == 'Leaf-Waste':
       return redirect(url_for('leaf'))
    if waste == 'Metal-Cans':
       return redirect(url_for('metalcans'))
    if waste == 'Paper-Waste':
       return redirect(url_for('paperwaste'))
    if waste == 'Plastic-Bags':
       return redirect(url_for('plasticbags'))
    if waste == 'Plastic-Bottles':
       return redirect(url_for('plasticbottles'))
    if waste == 'Wood-Waste':
       return redirect(url_for('woodwaste'))
  
    return waste


    
    


    
    
    

def get_aqi(api_key, city):
 url = f'https://api.airvisual.com/v2/city?city={city}&state=your_state&country=your_country&key={api_key}'
 response = requests.get(url)
 data = response.json()['data']['current']['pollution']
 aqi = data['aqius']
 return aqi
SUPPORTED_CITIES = ['city1', 'city2', 'city3']

def save_trends():
    dd = pd.read_csv('city_day.csv')
    dd_col = ['NO', 'NO2', 'NOx', 'NH3', 'CO','SO2', 'O3']
    for i in dd_col:
        a = dd[i].mean()
        dd[i].replace(np.nan , a,inplace =  True)
    Ahemdabad= dd.loc[dd['City'] == 'Ahmedabad']
    Ahemdabad.head(10)
    Ahemdabad['Date']= pd.to_datetime(Ahemdabad['Date']) 
    c =  ['City' , 'PM2.5' , 'PM10','Benzene' , 'Toluene', 'Xylene'  ,'AQI' ,'AQI_Bucket']
    Ahemdabad.drop(c, axis=1, inplace=True)
    Ahemdabad = Ahemdabad.sort_values('Date')
    Ahemdabad= Ahemdabad.reset_index()
    Ahemdabad = Ahemdabad.set_index('Date')

    # Considering the pollutant SO2 
    SO2 = Ahemdabad['SO2'].resample('2W').mean()

    # SAMIRA Model for SO2
    p = d = q = range(1, 4)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    mod = sm.tsa.statespace.SARIMAX(SO2,
                                    order=(2, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    # Forecasting for next 3 Years
    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    ax = SO2.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('SO2 Concentration')
    plt.legend()
    plt.plot()
    if os.path.exists('static/so2.png'):
            os.remove('static/so2.png')
    plt.savefig('static/so2.png')
    plt.close()


    ###############################################################################################

    # Considering the pollutant NO 
    NO = Ahemdabad['NO'].resample('2W').mean()
    # SAMIRA Model for NO
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    mod = sm.tsa.statespace.SARIMAX(NO,
                                    order=(2, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    # Forecasting for next 3 Years
    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    ax = NO.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('NO Concentration')
    plt.legend()
    plt.plot()
    if os.path.exists('static/NO.png'):
            os.remove('static/NO.png')
    plt.savefig('static/NO.png')
    plt.close()

    ###############################################################################################

     # Considering the pollutant NOx 
    NOx = Ahemdabad['NOx'].resample('2W').mean()

    # SAMIRA Model for NOx
    p = d = q = range(1, 4)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    mod = sm.tsa.statespace.SARIMAX(NOx,
                                    order=(2, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    # Forecasting for next 3 Years
    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    ax = NOx.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('NOx Concentration')
    plt.legend()
    plt.plot()
    if os.path.exists('static/NOx.png'):
            os.remove('static/NOx.png')
    plt.savefig('static/NOx.png')
    plt.close()

    # ###############################################################################################

    # # Considering the pollutant CO 
    # CO = Ahemdabad['CO'].resample('2W').mean()
    # # SAMIRA Model for CO
    # p = d = q = range(0, 3)
    # pdq = list(itertools.product(p, d, q))
    # seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    # mod = sm.tsa.statespace.SARIMAX(CO,
    #                                 order=(2, 1, 1),
    #                                 seasonal_order=(1, 1, 0, 12),
    #                                 enforce_stationarity=False,
    #                                 enforce_invertibility=False)
    # results = mod.fit()
    # # Forecasting for next 3 Years
    # pred_uc = results.get_forecast(steps=70)
    # pred_ci = pred_uc.conf_int()
    # ax = CO.plot(label='observed', figsize=(14, 7))
    # pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    # ax.fill_between(pred_ci.index,
    #                 pred_ci.iloc[:, 0],
    #                 pred_ci.iloc[:, 1], color='k', alpha=.25)
    # ax.set_xlabel('Date')
    # ax.set_ylabel('CO Concentration')
    # plt.legend()
    # plt.plot()
    # if os.path.exists('static/CO.png'):
    #         os.remove('static/CO.png')
    # plt.savefig('static/CO.png')
    # plt.close()

    ###############################################################################################

    # Considering the pollutant CO 
    O3 = Ahemdabad['O3'].resample('2W').mean()
    # SAMIRA Model for CO
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    mod = sm.tsa.statespace.SARIMAX(O3,
                                    order=(2, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    # Forecasting for next 3 Years
    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    ax = O3.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('O3 Concentration')
    plt.legend()
    plt.plot()
    if os.path.exists('static/O3.png'):
            os.remove('static/O3.png')
    plt.savefig('static/O3.png')
    plt.close()


    #############################################################################################

    # Considering the pollutant CO 
    NO2 = Ahemdabad['NO2'].resample('2W').mean()
    # SAMIRA Model for CO
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    mod = sm.tsa.statespace.SARIMAX(NO2,
                                    order=(2, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    # Forecasting for next 3 Years
    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    ax = NO2.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('NO2 Concentration')
    plt.legend()
    plt.plot()
    if os.path.exists('static/NO2.png'):
            os.remove('static/NO2.png')
    plt.savefig('static/NO2.png')
    plt.close()


@app.route('/')
def home(): 
    return render_template('home.html')

@app.route("/selected-value", methods=["POST"])
def handle_selected_value():
  data = request.get_json()
  selected_value = data["selectedValue"]
  print(selected_value)
  district_stats(selected_value)
  return jsonify({"success": True})

@app.route("/trends")
def trends():
    save_trends()
    return render_template('trend.html')

@app.route("/aqi")
def aqi():
    return render_template('aqi.html')

@app.route('/aqi_pred', methods = ['POST'])
def aqi_pred():
 if request.method == 'POST':
    pm25 = request.form.get['pm25']
    pm1 = request.form.get['pm1']
    no = request.form.get['no']
    nh = request.form.get['nh']
    co = request.form.get['co']
    so = request.form.get['so']
    o33 = request.form.get['o33']

    print(pm25, pm1, no, nh, co, so, o33)
    input = pd.DataFrame([[pm25, pm1, no, nh, co, so, o33]], columns=["PM2.5", "PM10", "NO2", "NH3", "CO", "SO2"])
    pred = pred_mod.predict(input)
    return str(np.round(pred,2))

@app.route('/recycling_guide', methods=['POST','GET', 'PUT'])

@app.route('/recycling-guide', methods=['GET', 'POST'])
def recycling_guide():
    # dropdown_value = request.form["selected_value"]
    display_stats()
    # if request.method == 'POST':
    #     city = request.form['city']
    #     aqi = get_aqi(API_KEY, city)
    #     return render_template('analysis.html', cities=SUPPORTED_CITIES, aqi=aqi)
    # else:
    return render_template('analysis.html')
     
    #   if request.method == 'POST':
    # city = request.form.get('city')
    #  url = API_ENDPOINT + 'city'
    #  params = {'city': 'city', 'state': 'your_state_here', 'country': 'your_country_here', 'key': AIRVISUAL_API_KEY}
    #  response = requests.get(url, params=params).json()
    #  city_name = response['data']['city']
    #  aqi = response['data']['current']['pollution']['aqius']
    #  return render_template('analysis.html', city_name=city_name, aqi=aqi)
    #  if request.method == 'POST':
    #     city = request.form.get('city')
    #     api_key = 'e84d5c69-af7c-49e3-b604-ad534ca5c7fe'  # Replace with your own AirVisual API key
    #         ##############################

            
    #     # Make API request to get AQI data for selected city
    #     response = requests.get(API_ENDPOINT, params={'city': city, 'key': api_key})
    #     data = response.json()['data']
    #     print(data)

    #     # Parse AQI data from response
    #     aqi = data['pollution']['aqius']
    #     aqi_desc = data['pollution']['mainus']
    #     temperature = data['weather']['tp']
    #     humidity = data['weather']['hu']

    #     return render_template('analysis.html', city=city, aqi=aqi, aqi_desc=aqi_desc,
    #                            temperature=temperature, humidity=humidity)
    #  else:
    #     return render_template('analysis.html')
    # return render_template('analysis.html')

    




@app.route('/types-of-waste',methods=['GET'])
def types_of_waste():
    return render_template('wasteclssi.html')
    

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        #save the file ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        # make prediction
        preds = model_predict(file_path,model)

        return preds
    return None

@app.route('/ewaste')
def ewaste():
    return render_template('e-waste.txt')

app.route('/food')
def food():
    return render_template('foodwaste.html')

@app.route('/leaf')
def leaf():
    return render_template('leaf.txt')

@app.route('/metalcans')
def metalcans():
    return render_template('metalcans.txt')

@app.route('/paperwaste')
def paperwaste():
    return render_template('paperwaste.txt')

@app.route('/plasticbags')
def plasticbags():
    return render_template('plasticbags.txt')

@app.route('/plastibottles')
def plasticbottles():
    return render_template('plasticbottles.txt')


@app.route('/woodwaste')
def woodwaste():
    return render_template('woodwaste.txt')

    

    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    
    #app.run(host="192.168.29.186", port=800, debug=False)

app.run()







