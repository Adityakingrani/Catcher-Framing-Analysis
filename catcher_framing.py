import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from pathlib import Path


def load_and_preprocess_data(filename):
    columns = [
        'CATCHER_ID', 'GAME_YEAR', 'PLATELOCHEIGHT', 'PLATELOCSIDE',
        'BALLS', 'STRIKES', 'BATTERSIDE', 'PITCHERTHROWS', 'PITCHCALL', 'PITCH_ID'
    ]
    data = pd.read_csv(filename, usecols=columns)

    data.rename(columns={'GAME_YEAR': 'Year'}, inplace=True)

    data['PLATELOCHEIGHT'].fillna(data['PLATELOCHEIGHT'].mean(), inplace=True)
    data['PLATELOCSIDE'].fillna(data['PLATELOCSIDE'].mean(), inplace=True)
    data['BALLS'].fillna(data['BALLS'].mode()[0], inplace=True)
    data['STRIKES'].fillna(data['STRIKES'].mode()[0], inplace=True)
    data['BATTERSIDE'].fillna('Unknown', inplace=True)
    data['PITCHERTHROWS'].fillna('Unknown', inplace=True)
    data['PITCHCALL'].fillna(data['PITCHCALL'].mode()[0], inplace=True)

    data['BATTERSIDE'] = data['BATTERSIDE'].map(
        {'Left': 0, 'Right': 1, 'Unknown': 2})
    data['PITCHERTHROWS'] = data['PITCHERTHROWS'].map(
        {'Left': 0, 'Right': 1, 'Unknown': 2})
    data['PITCHCALL'] = data['PITCHCALL'].map(
        {'BallCalled': 0, 'StrikeCalled': 1})

    X = data[['PLATELOCHEIGHT', 'PLATELOCSIDE', 'BALLS',
              'STRIKES', 'BATTERSIDE', 'PITCHERTHROWS']]
    y = data['PITCHCALL']

    return X, y, data


def train_model(X, y):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', LogisticRegression(max_iter=200))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    return pipeline


def apply_model_to_new_data(model, new_data_file, output_file):
    new_data = pd.read_csv(new_data_file)

    new_data['BATTERSIDE'] = new_data['BATTERSIDE'].map(
        {'Left': 0, 'Right': 1, 'Unknown': 2})
    new_data['PITCHERTHROWS'] = new_data['PITCHERTHROWS'].map(
        {'Left': 0, 'Right': 1, 'Unknown': 2})

    X_new = new_data[['PLATELOCHEIGHT', 'PLATELOCSIDE',
                      'BALLS', 'STRIKES', 'BATTERSIDE', 'PITCHERTHROWS']]
    new_data['Predicted_Strike_Prob'] = model.predict_proba(X_new)[:, 1]

    results = new_data.groupby(['CATCHER_ID', 'Year']).agg(
        Opportunities=('PITCH_ID', 'count'),
        Actual_Called_Strikes=('PITCHCALL', lambda x: (x == 1).sum()),
        Predicted_Called_Strikes=('Predicted_Strike_Prob', 'sum')
    ).reset_index()

    results['Called_Strikes_Added'] = (
        results['Actual_Called_Strikes'] - results['Predicted_Called_Strikes']).round(3)
    results['Called_Strikes_Added_Per_100'] = (
        (results['Called_Strikes_Added'] / results['Opportunities']) * 100).round(3)

    results[['CATCHER_ID', 'Year', 'Opportunities', 'Actual_Called_Strikes',
             'Called_Strikes_Added', 'Called_Strikes_Added_Per_100']].to_csv(output_file, index=False)


def main():
    try:
        if not Path('new_data.csv').exists():
            print("Error: new_data.csv not found in the current directory")
            return

        X, y, data = load_and_preprocess_data('ML_TAKES_ENCODED.csv')
        model = train_model(X, y)
        apply_model_to_new_data(model, 'new_data.csv', 'new_output.csv')

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
