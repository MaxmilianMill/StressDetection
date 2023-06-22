import pickle
import sklearn
import pandas as pd


def normalize(request: dict):
    '''
    normalize input values of the request
    '''

    # import min and max values of features
    min_max = pd.read_csv('min_max_features')
    min_max.drop('Unnamed: 0', axis=1, inplace=True)

    min_max_dict = dict()

    for feature in min_max:
        min_max_dict[feature] = dict(
            min=min_max.loc[:, feature][0], max=min_max.loc[:, feature][1])

    print(min_max_dict['TEMP']['min'])

    for feature, value in request.items():
        request[feature] = (value-min_max_dict[feature]['min']) / \
            (min_max_dict[feature]['max']-min_max_dict[feature]['min'])

    print(request)
    return [[request[feature] for feature in request.keys()]]


def ml_predict(request: dict):

    x_normalized = normalize(request=request)

    model = pickle.load(open('model.pkl', 'rb'))

    return model.predict(x_normalized)
