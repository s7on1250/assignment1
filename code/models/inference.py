import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from code.models.train import preprocess

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('models/standart_scalers.pkl', 'rb') as f:
    standart_scalers = pickle.load(f)
with open('models/min_max_scalers.pkl', 'rb') as f:
    min_max_scalers = pickle.load(f)
with open('models/classes_for_min_max_scale_0.pkl', 'rb') as f:
    classes_for_min_max_scale_0 = pickle.load(f)
with open('models/classes_for_min_max_scale_1.pkl', 'rb') as f:
    classes_for_min_max_scale_1 = pickle.load(f)


def preprocess_input(input_data):
    columns = pd.read_csv('data/cleaned.csv').columns
    dict_columns = dict()
    for i, c in enumerate(input_data.columns):
        dict_columns[c] = columns[i]
    input_data.rename(columns=dict_columns,inplace=True)
    input_data = preprocess(
        input_data,
        label_encoders,
        standart_scalers,
        min_max_scalers,
        classes_for_min_max_scale_0,
        classes_for_min_max_scale_1,
        train=False
    )
    return input_data


def inference(data):
    df = pd.DataFrame([data])
    preprocessed_data = preprocess_input(df)
    prediction = model.predict(preprocessed_data)
    return prediction[0]
