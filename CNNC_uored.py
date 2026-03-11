import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import yaml

from models.Networks import Network_bearing, Network_fan
from datasets.load_bearing_data import ReadDZLRSB, ReadMFPT, ReadUOTTAWA  # kept for original mode compatibility
from datasets.load_fan_data import ReadMIMII
from utils.CalIndex import cal_index
from utils.CreateLogger import create_logger
from utils.DictObj import DictObj
from utils.TuneReport import GenReport

from datasets.uored_condition_bridge import build_uored_condition_loaders
from utils.preds_export import collect_batch_meta, write_lightweight_preds_csv


def load_configs(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    configs = DictObj(cfg)
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
    return configs


class CenterLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes
        self.feat_dim = configs.dim_feature
        self.device = configs.device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size
        return loss


class CNNC(nn.Module):
    def __init__(self, configs):
        super().__init__()
        configs.num_domains = len(configs.datasets_src)
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type
        self.num_domains = configs.num_domains

        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.lr_cent = configs.lr_cent
        self.batch_size = configs.batch_size
        self.use_learning_rate_sheduler = configs.use_learning_rate_sheduler

        if self.dataset_type in ('bearing', 'uored'):
            self.model = Network_bearing(configs).to(self.device)
        elif self.dataset_type == 'fan':
            self.model = Network_fan(configs).to(self.device)
        else:
            raise ValueError('The dataset_type should be bearing, uored or fan!')

        if bool(getattr(configs, 'infer_dim_feature_from_input', False)):
            with torch.no_grad():
                dummy = torch.zeros(1, 1, int(configs.data_length), device=self.device)
                feat, _ = self.model(dummy)
                configs.dim_feature = int(feat.shape[1])

        self.center_loss = CenterLoss(configs).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)
        self.optimizer_cent = torch.optim.Adam(list(self.center_loss.parameters()), lr=self.lr_cent)
        self.cl_loss = nn.CrossEntropyLoss(reduction='none')

    def adjust_learning_rate(self, step):
        lr = self.lr
        if self.configs.cos:
            lr *= 0.5 * (1.0 + math.cos(math.pi * step / self.steps))
        else:
            for milestone in self.configs.schedule:
                lr *= 0.5 if step >= milestone else 1.0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, minibatches):
        x = torch.cat([batch[0] for batch in minibatches])
        labels = torch.cat([batch[1] for batch in minibatches])
        x = x.to(self.device)
        labels = labels.to(self.device)

        feature_vectors, logits = self.model(x)
        cl_loss = torch.mean(self.cl_loss(logits, labels))
        proto_loss = 0.5 * self.center_loss(feature_vectors, labels)
        loss = cl_loss + proto_loss

        self.optimizer.zero_grad()
        self.optimizer_cent.zero_grad()
        loss.backward()
        self.optimizer.step()
        for param in self.center_loss.parameters():
            param.grad.data *= (1.0 / 0.5)
        self.optimizer_cent.step()

        return {
            'cl': cl_loss.detach().cpu().item(),
            'proto': proto_loss.detach().cpu().item(),
        }

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)
        loss_acc_result = {'loss_proto': [], 'loss_cl': [], 'acces': []}

        for step in range(1, self.steps + 1):
            self.train()
            self.current_step = step
            self.logger.info('================Step {}================'.format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)
            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            loss_acc_result['loss_cl'].append(losses['cl'])
            loss_acc_result['loss_proto'].append(losses['proto'])
            self.logger.info('loss_cl_train: \t {loss_cl: .4f} \t loss_proto_train: \t {loss_proto: .4f} '.format(**losses))

            if step % self.checkpoint_freq == 0 or step == self.steps:
                acc_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)
                self.logger.info('tgt_train_acc: \t {src_train_acc: .4f}'.format(src_train_acc=acc_results[0]))
                for i in range(1, len(acc_results)):
                    self.logger.info('src_train_acc: \t {tgt_train_acc: .4f}'.format(tgt_train_acc=acc_results[i]))

        return loss_acc_result

    def test_model(self, loaders, export_pred_paths=None, loader_names=None, split_name=None):
        self.eval()
        acc_results = []
        export_pred_paths = export_pred_paths or [None] * len(loaders)
        loader_names = loader_names or [f'loader_{i}' for i in range(len(loaders))]

        for i, the_loader in enumerate(loaders):
            y_pred_lst = []
            y_true_lst = []
            records = []
            for batched_data in the_loader:
                if isinstance(batched_data, (list, tuple)) and len(batched_data) == 3:
                    x, label_fault, meta = batched_data
                else:
                    x, label_fault = batched_data
                    meta = None
                x = x.to(self.device)
                label_fault = label_fault.to(self.device)
                label_pred = self.predict(x)
                y_pred_np = label_pred.detach().cpu().numpy()
                y_true_np = label_fault.detach().cpu().numpy()
                y_pred_lst.extend(y_pred_np.tolist())
                y_true_lst.extend(y_true_np.tolist())

                if meta is not None:
                    meta_records = collect_batch_meta(meta)
                    if len(meta_records) != len(y_pred_np):
                        raise RuntimeError(f'Meta batch length mismatch: {len(meta_records)} vs {len(y_pred_np)}')
                    for meta_i, yt, yp in zip(meta_records, y_true_np.tolist(), y_pred_np.tolist()):
                        records.append({
                            'sample_id': meta_i.get('sample_id'),
                            'file_id': meta_i.get('file_id'),
                            'condition_id': meta_i.get('condition_id'),
                            'y_true': int(yt),
                            'y_pred': int(yp),
                            'method': 'CNNC',
                            'run_id': getattr(self.configs, 'run_id', ''),
                            'domain_split': split_name or '',
                            'loader_name': loader_names[i],
                        })

            acc_i, _, _, _ = cal_index(y_true_lst, y_pred_lst)
            acc_results.append(acc_i)
            out_csv = export_pred_paths[i] if i < len(export_pred_paths) else None
            if out_csv and records:
                write_lightweight_preds_csv(records, out_csv)
        self.train()
        return acc_results

    def predict(self, x):
        _, logits = self.model(x)
        return torch.max(logits, dim=1)[1]


def build_loaders_for_configs(configs):
    if bool(getattr(configs, 'use_uored_bridge', False)):
        train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names = build_uored_condition_loaders(configs)
        configs.datasets_tgt = target_names
        configs.datasets_src = source_names
        return train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names

    if configs.dataset_type == 'bearing':
        datasets_list = ['CWRU', 'UOTTAWA', 'MFPT', 'DZLRSB']
    elif configs.dataset_type == 'fan':
        configs.num_classes = 2
        configs.batch_size = 32
        configs.steps = 100
        configs.dim_feature = 640
        if configs.fan_section == 'sec00':
            datasets_list = ['W', 'X', 'Y', 'Z']
            section = '00'
        elif configs.fan_section == 'sec01':
            datasets_list = ['A', 'B', 'C']
            section = '01'
        else:
            datasets_list = ['L1', 'L2', 'L3', 'L4']
            section = '02'
    else:
        raise ValueError('The dataset_type should be bearing, uored or fan!')

    idx = 0
    dataset_idx = list(range(len(datasets_list)))
    tgt_idx = [idx]
    src_idx = [i for i in dataset_idx if i not in tgt_idx]
    datasets_tgt = [datasets_list[i] for i in tgt_idx]
    datasets_src = [datasets_list[i] for i in src_idx]
    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src

    if configs.dataset_type == 'bearing':
        datasets_object_src = [eval('Read' + i + '(configs)') for i in datasets_src]
        datasets_object_tgt = [eval('Read' + i + '(configs)') for i in datasets_tgt]
    else:
        datasets_object_src = [ReadMIMII(i, section=section, configs=configs) for i in datasets_src]
        datasets_object_tgt = [ReadMIMII(i, section=section, configs=configs) for i in datasets_tgt]

    train_test_loaders_src = [dataset.load_dataloaders() for dataset in datasets_object_src]
    train_loaders_src = [iter(train) for train, _ in train_test_loaders_src]
    test_loaders_src = [test for _, test in train_test_loaders_src]
    train_test_loaders_tgt = [dataset.load_dataloaders() for dataset in datasets_object_tgt]
    test_loaders_tgt = [test for _, test in train_test_loaders_tgt]
    return train_loaders_src, test_loaders_tgt, test_loaders_src, datasets_tgt, datasets_src


def parse_args():
    parser = argparse.ArgumentParser(description='CNNC with optional UORED bridge')
    parser.add_argument('--config', type=str, default=os.path.join(sys.path[0], 'config_files', 'CNNC_config.yaml'))
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_configs(args.config)
    train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names = build_loaders_for_configs(configs)
    train_minibatches_iterator = zip(*train_loaders_src)

    run_root = os.path.join(getattr(configs, 'output_root', 'Output/CNNC_UORED'), getattr(configs, 'run_id', str(time.time())[:10]))
    full_path_log = os.path.join(run_root, 'log_files')
    full_path_rep = os.path.join(run_root, 'TuneReport')
    pred_dir = os.path.join(run_root, 'artifacts')
    os.makedirs(full_path_log, exist_ok=True)
    os.makedirs(full_path_rep, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, 'log_file' + currtime))
    model = CNNC(configs)
    for k, v in sorted(vars(configs).items()):
        logger.info('\t{}: {}'.format(k, v))

    loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt + test_loaders_src, logger)
    loss_acc_result['loss_cl'] = np.array(loss_acc_result['loss_cl'])
    loss_acc_result['loss_proto'] = np.array(loss_acc_result['loss_proto'])
    loss_acc_result['acces'] = np.array(loss_acc_result['acces'])

    sio.savemat(os.path.join(full_path_log, 'loss_acc_result' + currtime + '.mat'), loss_acc_result)
    gen_report = GenReport(full_path_rep)
    gen_report.write_file(configs=configs, test_item=None, loss_acc_result=loss_acc_result)
    gen_report.save_file(currtime)

    if bool(getattr(configs, 'export_test_preds', True)):
        target_pred_paths = [os.path.join(pred_dir, f'test_preds__{name}.csv') for name in target_names]
        model.test_model(
            test_loaders_tgt,
            export_pred_paths=target_pred_paths,
            loader_names=target_names,
            split_name='target',
        )


if __name__ == '__main__':
    main()