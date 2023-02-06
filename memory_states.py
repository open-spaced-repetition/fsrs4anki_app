import numpy as np
from functools import partial

import pandas as pd


def predict_memory_states(my_collection, group):
    states = my_collection.states(*group.name)
    group['stability'] = float(states[0])
    group['difficulty'] = float(states[1])
    group['count'] = len(group)
    return pd.DataFrame({
        'r_history': [group.name[1]],
        't_history': [group.name[0]],
        'stability': [round(float(states[0]), 2)],
        'difficulty': [round(float(states[1]), 2)],
        'count': [len(group)]
        })


def get_my_memory_states(proj_dir, dataset, my_collection):
    prediction = dataset.groupby(by=['t_history', 'r_history']).progress_apply(
        partial(predict_memory_states, my_collection))
    prediction.reset_index(drop=True, inplace=True)
    prediction.sort_values(by=['r_history'], inplace=True)
    prediction.to_csv(proj_dir / "prediction.tsv", sep='\t', index=None)
    # print("prediction.tsv saved.")
    prediction['difficulty'] = prediction['difficulty'].map(lambda x: int(round(x)))
    difficulty_distribution = prediction.groupby(by=['difficulty'])['count'].sum() / prediction['count'].sum()
    # print(difficulty_distribution)
    difficulty_distribution_padding = np.zeros(10)
    for i in range(10):
        if i + 1 in difficulty_distribution.index:
            difficulty_distribution_padding[i] = difficulty_distribution.loc[i + 1]
    return difficulty_distribution_padding, difficulty_distribution
