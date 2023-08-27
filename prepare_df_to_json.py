import pandas as pd
import json

df = pd.read_csv(r'data/02324400.csv').iloc[-100:, ].reset_index(drop=True)

config = {
    'look_back': 10,
    'lead_time': 1,
    'feature_cols': ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'QObs(mm/d)'],
    'target_cols': ['QObs(mm/d)']
}

df = df[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)',
         'tmax(C)', 'tmin(C)', 'vp(Pa)', 'QObs(mm/d)', 'QObs(mm/d)']]
data_dict = df.T.to_dict()

results_dict = {'data': {'sample': data_dict, 'params': config}}
with open("data/test.json", "w") as f:
    json.dump(results_dict, f)
    print("已生成news_json.json文件...")

json_dict = json.load(open("data/test.json"))['data']
df = pd.DataFrame(json_dict)
sample_df = df[['sample']].dropna(axis=0)
sample_df_convert = pd.DataFrame(sample_df.to_dict()['sample']).T
param_df = df[['params']].dropna(axis=0)
param_dict = param_df.to_dict()['params']