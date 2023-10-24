import pandas as pd

from sklearn.linear_model import LogisticRegression


def fill_unknown_previous_rate(data: pd.DataFrame, mean=None) -> float:
    """Estimate unknown previous rate with mean value
    """
    data['Previous_Rate_Known'] = (data['Previous_Rate'] != -10)
    
    prev_rate = data['Previous_Rate'].copy()
    if mean:
        prev_rate_mean = mean
    else:
        prev_rate_mean = prev_rate[prev_rate != -10].mean()
    
    prev_rate[prev_rate == -10] = prev_rate_mean
    data['Previous_Rate'] = prev_rate
    
    return prev_rate_mean


def split_on_observations_nd_actions(data: pd.DataFrame):
    observations = data[['Tier', 'FICO', 'Term', 'Amount', 'Previous_Rate', 'Competition_rate',
                         'Cost_Funds', 'Partner Bin', 'Car_Type_N', 'Car_Type_R', 'Car_Type_U']]
    actions = data['Rate']

    observations = observations.values
    actions = actions.values.reshape(-1, 1)

    return observations, actions


def get_acceptance_model(data: pd.DataFrame):
    X = data[['Tier', 'FICO', 'Term', 'Amount', 'Previous_Rate', 'Competition_rate',
              'Cost_Funds', 'Partner Bin', 'Car_Type_N', 'Car_Type_R', 'Car_Type_U', 'Rate']].values
    y = data['Accept'].values

    model = LogisticRegression(max_iter=300, penalty=None, fit_intercept=True, 
                               multi_class='ovr', random_state=42, n_jobs=4).fit(X, y)
    
    return model
