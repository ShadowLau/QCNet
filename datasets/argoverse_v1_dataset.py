# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib import request

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data import extract_tar
from tqdm import tqdm


class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 sample_interval = 1,
                 dim: int = 3,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 30,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True) -> None:
        root = os.path.expanduser(os.path.normpath(root))
        if not os.path.isdir(root):
            os.makedirs(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split

        if raw_dir is None:
            raw_dir = os.path.join(root, split, 'data')
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = os.listdir(self._raw_dir)
                self._raw_file_names = self._raw_file_names[::sample_interval]
            else:
                self._raw_file_names = []
        else:
            raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = os.listdir(self._raw_dir)
                self._raw_file_names = self._raw_file_names[::sample_interval]
            else:
                self._raw_file_names = []

        if processed_dir is None:
            processed_dir = os.path.join(root, split, 'processed')
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = os.listdir(self._processed_dir)
                self._processed_file_names = self._processed_file_names[::sample_interval]
            else:
                self._processed_file_names = []
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = os.listdir(self._processed_dir)
                self._processed_file_names = self._processed_file_names[::sample_interval]
            else:
                self._processed_file_names = []

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'

        train_samples = (205942-1) // sample_interval + 1
        val_samples = (39472-1) // sample_interval + 1
        self._num_samples = {
            'train': train_samples,
            'val': val_samples,
            'test': 78143,
        }[split]
        self._agent_types = ['vehicle', 'unknown']
        self._agent_categories = ['OTHERS', 'AV', 'AGENT']
        super(ArgoverseV1Dataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)


    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def download(self) -> None:
        if not os.path.isfile(os.path.join(self.root, f'{self.split}.tar')):
            print(f'Downloading {self._url}', file=sys.stderr)
            request.urlretrieve(self._url, os.path.join(self.root, f'{self.split}.tar'))
        if os.path.isdir(os.path.join(self.root, self.split)):
            shutil.rmtree(os.path.join(self.root, self.split))
        if os.path.isdir(self.raw_dir):
            shutil.rmtree(self.raw_dir)
        os.makedirs(self.raw_dir)
        extract_tar(path=os.path.join(self.root, f'{self.split}.tar'), folder=self.raw_dir, mode='r')
        self._raw_file_names = [name for name in os.listdir(os.path.join(self.raw_dir, self.split)) if
                                os.path.isdir(os.path.join(self.raw_dir, self.split, name))]
        for raw_file_name in self.raw_file_names:
            shutil.move(os.path.join(self.raw_dir, self.split, raw_file_name), self.raw_dir)
        os.rmdir(os.path.join(self.raw_dir, self.split))

    def process(self) -> None:
        for raw_file_name in tqdm(self.raw_file_names):
            df = pd.read_csv(os.path.join(self.raw_dir, raw_file_name))

            timestamp = list(df['TIMESTAMP'].unique())
            time_map = {timestamp[i]: i for i in range(len(timestamp))}
            timestep = [time_map[x] for x in list(df['TIMESTAMP'])]
            df['timestep'] = timestep

            obj_types = list(df['OBJECT_TYPE'])
            object_category = [2 if x == 'AGENT' else 1 if x=='AV' else 0 for x in obj_types]
            df['object_category'] = object_category

            object_type = ['vehicle' if x in ('AGENT', 'AV') else 'unknown' for x in obj_types]
            df['object_type'] = object_type

            all_track_id = list(df['TRACK_ID'])
            track_id_ori = list(df['TRACK_ID'].unique())
            id_map = {track_id_ori[i]: i for i in range(len(track_id_ori))}
            track_id = [id_map[all_track_id[i]] if obj_types[i]!='AV' else 'AV' for i in range(len(all_track_id))]
            df['track_id'] = track_id

            name = raw_file_name.split('.')[0]
            data = dict()
            data['scenario_id'] = name
            data['city'] = df['CITY_NAME'].values[0]
            data['agent'] = self.get_agent_features(df)
            with open(os.path.join(self.processed_dir, f'{name}.pkl'), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
            historical_df = df[df['timestep'] < self.num_historical_steps]
            agent_ids = list(historical_df['track_id'].unique())
            df = df[df['track_id'].isin(agent_ids)]
        else:
            agent_ids = list(df['track_id'].unique())

        num_agents = len(agent_ids)
        av_idx = agent_ids.index('AV')

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)

        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df['timestep'].values
            agent_timestamp = track_df['TIMESTAMP'].values

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            if self.vector_repr:  # a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1: self.num_historical_steps] = (
                        valid_mask[agent_idx, :self.num_historical_steps - 1] &
                        valid_mask[agent_idx, 1: self.num_historical_steps])
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(track_df['object_type'].values[0])
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

        if self.split == 'test':
            predict_mask[current_valid_mask
                            | (agent_category == 2)
                            | (agent_category == 3), self.num_historical_steps:] = True

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


    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        with open(self.processed_paths[idx], 'rb') as handle:
            return HeteroData(pickle.load(handle))

    def _download(self) -> None:
        # if complete raw/processed files exist, skip downloading
        if ((os.path.isdir(self.raw_dir) and len(self.raw_file_names) == len(self)) or
                (os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self))):
            return
        self._processed_file_names = []
        print("*" * 80)
        print("downloading")
        # self.download()

    def _process(self) -> None:
        # if complete processed files exist, skip processing
        if os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self):
            return
        print('Processing...', file=sys.stderr)
        if os.path.isdir(self.processed_dir):
            for name in os.listdir(self.processed_dir):
                if name.endswith(('pkl', 'pickle')):
                    os.remove(os.path.join(self.processed_dir, name))
        else:
            os.makedirs(self.processed_dir)
        self._processed_file_names = [f'{raw_file_name}.pkl' for raw_file_name in self.raw_file_names]
        self.process()
        print('Done!', file=sys.stderr)