
import numpy as np
import pandas as pd
import torch
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import os
import pickle


def get_agent_features(df: pd.DataFrame) -> Dict[str, Any]:
    predict_unseen_agents = False
    num_historical_steps = 20
    num_future_steps = 30
    num_steps = num_historical_steps + num_future_steps
    dim = 3
    vector_repr = True
    _agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
                                'construction', 'riderless_bicycle', 'unknown']
    _agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
    split = 'test'


    if not predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
        historical_df = df[df['timestep'] < num_historical_steps]
        agent_ids = list(historical_df['track_id'].unique())
        df = df[df['track_id'].isin(agent_ids)]
    else:
        agent_ids = list(df['track_id'].unique())

    num_agents = len(agent_ids)
    av_idx = agent_ids.index('AV')

    # initialization
    valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
    predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    agent_id: List[Optional[str]] = [None] * num_agents
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)
    position = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    velocity = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)

    for track_id, track_df in df.groupby('track_id'):
        agent_idx = agent_ids.index(track_id)
        agent_steps = track_df['timestep'].values
        agent_timestamp = track_df['TIMESTAMP'].values

        valid_mask[agent_idx, agent_steps] = True
        current_valid_mask[agent_idx] = valid_mask[agent_idx, num_historical_steps - 1]
        predict_mask[agent_idx, agent_steps] = True
        if vector_repr:  # a time step t is valid only when both t and t-1 are valid
            valid_mask[agent_idx, 1: num_historical_steps] = (
                    valid_mask[agent_idx, :num_historical_steps - 1] &
                    valid_mask[agent_idx, 1: num_historical_steps])
            valid_mask[agent_idx, 0] = False
        predict_mask[agent_idx, :num_historical_steps] = False
        if not current_valid_mask[agent_idx]:
            predict_mask[agent_idx, num_historical_steps:] = False

        agent_id[agent_idx] = track_id
        agent_type[agent_idx] = _agent_types.index('vehicle')
        agent_category[agent_idx] = track_df['object_category'].values[0]
        position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['X'].values, track_df['Y'].values],
                                                                         axis=-1)).float()
        
        x = track_df['X'].values
        y = track_df['Y'].values
        if len(x) < 2:
            continue

        diff_x = x[1:] - x[:-1]
        diff_y = y[1:] - y[:-1]
        diff_t = agent_timestamp[1:] - agent_timestamp[:-1]
        diff_x = np.insert(diff_x, 0, diff_x[0])
        diff_y = np.insert(diff_y, 0, diff_y[0])
        diff_t = np.insert(diff_t, 0, diff_t[0])
        agent_heading = np.arctan2(diff_y, diff_x)

        velocity_x = diff_x / diff_t
        velocity_y = diff_y / diff_t

        heading[agent_idx, agent_steps] = torch.from_numpy(agent_heading).float()
        velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([velocity_x, velocity_y], 
                                                                         axis=-1)).float()

    if split == 'test':
        predict_mask[current_valid_mask
                        | (agent_category == 2)
                        | (agent_category == 3), num_historical_steps:] = True

    return {
        'num_nodes': num_agents,
        'av_index': av_idx,
        'valid_mask': valid_mask,
        'predict_mask': predict_mask,
        'id': agent_id,
        'type': agent_type,
        'category': agent_category,
        'position': position,
        'heading': heading,
        'velocity': velocity,
    }



if __name__ == '__main__':
    av1_file = '/data/xiaodliu/av1/motion/train/data/100000.csv'

    df1 = pd.read_csv(av1_file)

    timestamp = list(df1['TIMESTAMP'].unique())
    time_map = {timestamp[i]: i for i in range(len(timestamp))}
    timestep = [time_map[x] for x in list(df1['TIMESTAMP'])]
    df1['timestep'] = timestep

    obj_types = list(df1['OBJECT_TYPE'])
    object_category = [2 if x == 'AGENT' else 1 for x in obj_types]
    df1['object_category'] = object_category

    all_track_id = list(df1['TRACK_ID'])
    track_id_ori = list(df1['TRACK_ID'].unique())
    id_map = {track_id_ori[i]: i for i in range(len(track_id_ori))}
    track_id = [id_map[all_track_id[i]] if obj_types[i]!='AV' else 'AV' for i in range(len(all_track_id))]
    df1['track_id'] = track_id

    data = dict()
    data['scenario_id'] = '100000'
    data['city'] = df1['CITY_NAME'].values[0]
    data['agent'] = get_agent_features(df1)


