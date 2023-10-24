import numpy as np


def p_default(state):
    fico = state[:, 1]
    
    default_prob = fico.copy()
    default_prob[default_prob < 500] = 0.41
    default_prob[default_prob >= 750] = 0.01
    default_prob[default_prob >= 700] = 0.044
    default_prob[default_prob >= 650] = 0.089
    default_prob[default_prob >= 600] = 0.158
    default_prob[default_prob >= 550] = 0.225
    default_prob[default_prob >= 500] = 0.284

    return default_prob


def calc_accept_prob(action, state, model):
    if isinstance(action, (int, float)):
        action = np.array(action)

    action = action.reshape(-1, 1)
    
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    
    x = np.concatenate((state, action), axis=1)
    probs = model.predict_proba(x)[:, 1]
    
    return probs


def reward(action, state, model, risk_free=0.2, loss_ratio=0.5):
    if len(state.shape) == 1:
        state = state.reshape(1, -1)

    p_accept= calc_accept_prob(action, state, model)

    Sum_loan = state[:, 3]
    Term = state[:, 2] / 12

    p_return = p_default(state)

    loss_given_default = loss_ratio * Sum_loan
    action = action / 100
    reward = (p_accept \
        * (Sum_loan*p_return*((1+action)**Term-(1+risk_free)**Term)-(1-p_return) \
            * loss_given_default)) \
        / Sum_loan

    return reward
