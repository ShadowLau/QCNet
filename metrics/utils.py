# Copyright (c) 2023, Zikang Zhou. All rights reserved.
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

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_scatter import gather_csr
from torch_scatter import segment_csr

import h5py
import numpy as np



def topk(
        max_guesses: int,
        pred: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        joint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    max_guesses = min(max_guesses, pred.size(1))
    if max_guesses == pred.size(1):
        if prob is not None:
            prob = prob / prob.sum(dim=-1, keepdim=True)
        else:
            prob = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred, prob
    else:
        if prob is not None:
            if joint:
                if ptr is None:
                    inds_topk = torch.topk((prob / prob.sum(dim=-1, keepdim=True)).mean(dim=0, keepdim=True),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = inds_topk.repeat(pred.size(0), 1)
                else:
                    inds_topk = torch.topk(segment_csr(src=prob / prob.sum(dim=-1, keepdim=True), indptr=ptr,
                                                       reduce='mean'),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = gather_csr(src=inds_topk, indptr=ptr)
            else:
                inds_topk = torch.topk(prob, k=max_guesses, dim=-1, largest=True, sorted=True)[1]
            pred_topk = pred[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)
        else:
            pred_topk = pred[:, :max_guesses]
            prob_topk = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred_topk, prob_topk


def valid_filter(
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
                                                       torch.Tensor, torch.Tensor]:
    if valid_mask is None:
        valid_mask = target.new_ones(target.size()[:-1], dtype=torch.bool)
    if keep_invalid_final_step:
        filter_mask = valid_mask.any(dim=-1)
    else:
        filter_mask = valid_mask[:, -1]
    pred = pred[filter_mask]
    target = target[filter_mask]
    if prob is not None:
        prob = prob[filter_mask]
    valid_mask = valid_mask[filter_mask]
    if ptr is not None:
        num_nodes_batch = segment_csr(src=filter_mask.long(), indptr=ptr, reduce='sum')
        ptr = num_nodes_batch.new_zeros((num_nodes_batch.size(0) + 1,))
        torch.cumsum(num_nodes_batch, dim=0, out=ptr[1:])
    else:
        ptr = target.new_tensor([0, target.size(0)])
    return pred, target, prob, valid_mask, ptr



TYPE_LIST = Union[List[np.ndarray], np.ndarray]

def generate_forecasting_h5(
    data: Dict[int, TYPE_LIST],
    output_path: str,
    filename: str = "argoverse_forecasting_baseline",
    probabilities: Optional[Dict[int, List[float]]] = None,
) -> None:
    """
    Helper function to generate the result h5 file for argoverse forecasting challenge

    Args:
        data: a dictionary of trajectories, with the key being the sequence ID, and value being
              predicted trajectories for the sequence, stored in a (n,30,2) np.ndarray.
              "n" can be any number >=1. If probabilities are provided, the evaluation server
              will use the top-K most likely forecasts for any top-K metric. If probabilities
              are unavailable, the first-K trajectories will be evaluated instead. Each
              predicted trajectory should consist of 30 waypoints.
        output_path: path to the output directory to store the output h5 file
        filename: to be used as the name of the file
        probabilities (optional) : normalized probability for each trajectory
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hf = h5py.File(os.path.join(output_path, filename + ".h5"), "w")
    future_frames = 30
    d_all: List[np.ndarray] = []
    counter = 0
    for key, value in data.items():
        print("\r" + str(counter + 1) + "/" + str(len(data)), end="")

        if isinstance(value, List):
            value = np.array(value)
        assert value.shape[1:3] == (
            future_frames,
            2,
        ), f"ERROR: the data should be of shape (n,30,2), currently getting {value.shape}"

        n = value.shape[0]
        len_val = len(value)
        value = value.reshape(n * future_frames, 2)
        if probabilities is not None:
            assert key in probabilities.keys(), f"missing probabilities for sequence {key}"
            assert (
                len(probabilities[key]) == len_val
            ), f"mismatch sequence and probabilities len for {key}: {len(probabilities[key])} !== {len_val}"
            # assert np.isclose(np.sum(probabilities[key]), 1), "probabilities are not normalized"

            d = np.array(
                [
                    [
                        key,
                        np.float32(x),
                        np.float32(y),
                        probabilities[key][int(np.floor(i / future_frames))],
                    ]
                    for i, (x, y) in enumerate(value)
                ]
            )
        else:
            d = np.array([[key, np.float32(x), np.float32(y)] for x, y in value])

        d_all.append(d)
        counter += 1

    d_all = np.concatenate(d_all, 0)
    hf.create_dataset("argoverse_forecasting", data=d_all, compression="gzip", compression_opts=9)
    hf.close()
