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
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch import loggers as pl_loggers

from datamodules import ArgoverseV2DataModule, ArgoverseV1DataModule
from predictors import QCNet


if __name__ == '__main__':

    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='train')
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--no_map', action='store_true')
    parser.add_argument('--data_to_ram', action='store_true')
    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()

    args.T_max = args.max_epochs

    log_dir = f'exps/{args.exp_name}'
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    model = QCNet(**vars(args))
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
        'argoverse_v1': ArgoverseV1DataModule,
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs,
                         logger=tb_logger, check_val_every_n_epoch=8)
    trainer.fit(model, datamodule, ckpt_path=args.resume_path)
