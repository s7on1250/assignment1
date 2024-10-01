import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle


def preprocess(x_inp, label_encoders, standart_scalers, min_max_scalers, classes_for_min_max_scale_0,
               classes_for_min_max_scale_1, train=True):
    x = x_inp.copy()
    if 'Unnamed: 0' in x.columns:
        x = x.drop(['Unnamed: 0'], axis=1)
    if 'id' in x.columns:
        x = x.drop(['id'], axis=1)
    if train:
        label_encoders[0].fit(x.Gender)
        label_encoders[1].fit(x['Customer Type'])
        label_encoders[2].fit(x['Type of Travel'])
        label_encoders[3].fit(x['Class'])

        standart_scalers[0].fit(np.array(x.Age).reshape(-1, 1))
        standart_scalers[1].fit(np.array(x['Flight Distance']).reshape(-1, 1))
        standart_scalers[2].fit(np.array(x['Departure Delay in Minutes']).reshape(-1, 1))
        standart_scalers[3].fit(np.array(x['Arrival Delay in Minutes']).reshape(-1, 1))

        min_max_scalers[0].fit(np.array(x[classes_for_min_max_scale_0]).reshape(-1, 1))
        min_max_scalers[1].fit(np.array(x[classes_for_min_max_scale_1]).reshape(-1, 1))
    x.Gender = label_encoders[0].transform(x.Gender)
    x['Customer Type'] = label_encoders[1].transform(x['Customer Type'])
    x['Type of Travel'] = label_encoders[2].transform(x['Type of Travel'])
    x['Class'] = label_encoders[3].transform(x['Class'])
    x['Customer Type'] = x['Customer Type'].replace({'Loyal Customer': 0, 'disloyal Customer': 1})
    x.Age = standart_scalers[0].transform(np.array(x.Age).reshape(-1, 1))
    x['Flight Distance'] = standart_scalers[1].transform(np.array(x['Flight Distance']).reshape(-1, 1))
    x['Departure Delay in Minutes'] = standart_scalers[2].transform(
        np.array(x['Departure Delay in Minutes']).reshape(-1, 1))
    x['Arrival Delay in Minutes'] = standart_scalers[3].transform(
        np.array(x['Arrival Delay in Minutes']).reshape(-1, 1))

    for c in classes_for_min_max_scale_0:
        x[c] = min_max_scalers[0].transform(np.array(x[c]).reshape(-1, 1))
    for c in classes_for_min_max_scale_1:
        x[c] = min_max_scalers[1].transform(np.array(x[c]).reshape(-1, 1))
    return pd.DataFrame(x)


def main():
    df_train = pd.read_csv('code/datasets/train.csv')
    df_test = pd.read_csv('code/datasets/test.csv')
    x_train, x_test, y_train, y_test = df_train.drop('satisfaction', axis=1), df_test.drop('satisfaction', axis=1), \
    df_train['satisfaction'], df_test['satisfaction']

    label_encoders = [LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()]
    standart_scalers = [StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()]
    min_max_scalers = [MinMaxScaler(), MinMaxScaler()]
    classes_for_min_max_scale_0 = [c for c in x_train.columns if len(x_train[c].unique()) == 6]
    classes_for_min_max_scale_1 = [c for c in x_train.columns if len(x_train[c].unique()) == 5]

    x_train = preprocess(x_train, label_encoders, standart_scalers, min_max_scalers, classes_for_min_max_scale_0,
                         classes_for_min_max_scale_1)
    x_test = preprocess(x_test, label_encoders, standart_scalers, min_max_scalers, classes_for_min_max_scale_0,
                        classes_for_min_max_scale_1, train=False)

    model = CatBoostClassifier(iterations=100,
                               learning_rate=0.01,
                               depth=12,
                               allow_writing_files=False)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print(f1, accuracy)
    x_train.to_csv('data/cleaned.csv', index=False)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open('models/standart_scalers.pkl', 'wb') as f:
        pickle.dump(standart_scalers, f)
    with open('models/min_max_scalers.pkl', 'wb') as f:
        pickle.dump(min_max_scalers, f)
    with open('models/classes_for_min_max_scale_0.pkl', 'wb') as f:
        pickle.dump(classes_for_min_max_scale_0, f)
    with open('models/classes_for_min_max_scale_1.pkl', 'wb') as f:
        pickle.dump(classes_for_min_max_scale_1, f)


if __name__ == '__main__':
    main()
