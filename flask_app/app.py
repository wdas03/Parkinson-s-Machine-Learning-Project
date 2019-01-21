from flask import Flask, render_template, request
import json
import pickle
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, iqr

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
tree = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'tree_clf.pkl'), 'rb'))
    
def process_json_data(json):
    df = pd.DataFrame(columns=['Tremors', 'Gender', 'Sided', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time'])
    categorical_data = json[0]
    numerical_data = json[1:]
    
    idx = 0
    for row in numerical_data:
        for attrib in ["Hand", "Hold time", "Direction", "Latency time", "Flight time"]:
            df.loc[idx, attrib] = row[attrib]
        idx = idx + 1
        
    df.loc[:, 'Hold time'] = df[df['Hand'].isin(['L', 'R', 'S'])]['Hold time'].apply(pd.to_numeric, downcast='float', errors='coerce')
    df.loc[:, 'Latency time'] = df[df['Direction'].isin(['LL', 'LR', 'RR', 'RL'])]['Latency time'].apply(pd.to_numeric, downcast='float', errors='coerce')
    df.loc[:, 'Flight time'] = df['Flight time'].apply(pd.to_numeric, downcast='float', errors='coerce')
    
    #df["Hold time"].fillna(df["Hold time"].mean(), inplace=True)
    #df["Latency time"].fillna(df["Latency time"].mean(), inplace=True)
    #df["Flight time"].fillna(df["Flight time"].mean(), inplace=True)
    
    df = df[(df['Hold time'] > 0) & (df['Latency time'] > 0) & (df['Hold time'] < 2000) & (df['Latency time'] < 2000)]
    
    hold_time_stats = df.groupby("Hand")["Hold time"].agg([ np.mean, np.median, np.max, np.min, np.var, iqr, skew, kurtosis ])
    latency_time_stats = df.groupby("Direction")["Latency time"].agg([ np.mean, np.median, np.max, np.min, np.var, iqr, skew, kurtosis ])
    flight_time_stats = df.groupby("Direction")["Flight time"].agg([ np.mean, np.median, np.max, np.min, np.var, iqr, skew, kurtosis ])
    
    hold_time_stats.fillna(hold_time_stats.mean(), inplace=True)
    latency_time_stats.fillna(latency_time_stats.mean(), inplace=True)
    
    """
    Columns to use:
    ['Tremors', 'Sided_None', 'Sided_Right', 'mean_L_hold_time',
       'mean_R_hold_time', 'mean_S_hold_time', 'median_L_hold_time',
       'median_R_hold_time', 'median_S_hold_time', 'amax_L_hold_time',
       'amax_S_hold_time', 'var_R_hold_time', 'std_S_hold_time',
       'skew_L_hold_time', 'skew_S_hold_time', 'amax_LL_latency_time',
       'amax_LR_latency_time', 'amax_RL_latency_time', 'amin_LL_latency_time',
       'amin_RL_latency_time', 'var_RL_latency_time', 'var_RR_latency_time',
       'std_RR_latency_time', 'kurtosis_RL_latency_time',
       'kurtosis_RR_latency_time']
    """
    variables = ["mean", "median", "amax", "amin", "var", "iqr", "skew", "kurtosis"]
    full_data_df = pd.DataFrame()
    data_types = []
    data_types.append(('hold_time', hold_time_stats))
    data_types.append(('latency_time', latency_time_stats))
    for data_type in data_types:  
        for stat in variables:
            name_to_append = data_type[0]
            data = data_type[1]
            values = data[stat]
            for idx, val in values.iteritems():
                column_name = stat + "_" + idx + "_" + name_to_append
                full_data_df.loc[0, column_name] = val
    
    optimal_columns = ['Tremors', 'Sided_None', 'mean_L_hold_time', 'mean_S_hold_time',
       'amax_L_hold_time', 'var_S_hold_time', 'iqr_S_hold_time',
       'skew_L_hold_time', 'skew_R_hold_time', 'skew_S_hold_time',
       'kurtosis_R_hold_time', 'mean_LR_latency_time', 'mean_RL_latency_time',
       'var_RR_latency_time', 'iqr_LR_latency_time', 'iqr_RR_latency_time',
       'skew_LR_latency_time']
        
    for column in ["Tremors", "Sided"]:
        full_data_df.loc[:, column] = categorical_data[column]
    
    full_data_df = pd.get_dummies(full_data_df, columns=['Sided'])
    full_data_df.fillna(full_data_df.mean(), inplace=True)
    print(full_data_df.sort_index(axis=1))
    X = full_data_df.loc[0, optimal_columns]
    print(df.groupby("Direction").last().loc["RR", "Latency time"])
    #X["std_RR_latency_time"]
    if categorical_data["Sided"] == "Left":
        X.loc["Sided_None"] = 0
    elif categorical_data["Sided"] == "Right":
        X.loc["Sided_None"] = 0
    
    X = X.replace({"Yes": 1, "No": 0})
    X.fillna(0, inplace=True)
    
    X_input = [X.values.tolist()]
    
    print(X_input)
    print(tree.predict(X_input))
    #print(tree.predict(X.tolist()))
    #y_pred = clf.predict(X)
    #print(y_pred)
    
        #print(values)
    # Unstack and disperse values into columns for hold and latency times
    #hold_time_stats = hold_time_stats.unstack()
    #print(hold_time_stats)
    #print(hold_time_stats.index.values[0][0] + "_" + hold_time_stats.index.values[0][1])
    #columns = [i[0] + "_" + i[1] for i in hold_time_stats.index.values]
    #print(columns)
    #print(hold_time_stats.unstack())
    #hold_time_stats.columns = ['_'.join(col).strip() + '_hold_time' for col in hold_time_stats.columns.values]
    #hold_time_stats['R_L_mean_difference'] = abs(hold_time_stats['mean_L_hold_time'] - hold_time_stats['mean_R_hold_time'])
    
    #latency_time_stats = latency_time_stats.unstack()
    #latency_time_stats.columns = ['_'.join(col).strip() + '_latency_time' for col in latency_time_stats.columns.values]
    #latency_time_stats['LL_RR_mean_difference'] = abs(latency_time_stats['mean_LL_latency_time'] - latency_time_stats['mean_RR_latency_time'])
    #latency_time_stats['LR_RL_mean_difference'] = abs(latency_time_stats['mean_LR_latency_time'] - latency_time_stats['mean_RL_latency_time'])
    #print(df)
    #print(df)
    #print(categorical_data)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        json_data = json.loads(request.data)
        process_json_data(json_data)
        #return render_template('results.html', data="what up")
        #print(request.get_json())
        return json.dumps({"status": "OK", "data": json_data})

if __name__ == '__main__':
    app.run()
