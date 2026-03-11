import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.Networks import Network_bearing, Network_fan
from utils.CalIndex import cal_index
from utils.CreateLogger import create_logger
from utils.DictObj import DictObj

from datasets.uored_condition_bridge import build_uored_condition_loaders
from utils.preds_export import collect_batch_meta, write_lightweight_preds_csv

# optional / lazy imports for original benchmark compatibility
ReadCWRU = None
ReadDZLRSB = None
ReadJNU = None
ReadPU = None
ReadMFPT = None
ReadUOTTAWA = None
ReadMIMII = None
GenReport = None


def load_configs(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    configs = DictObj(cfg)
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = "cuda"
    return configs


class DomainContrastiveLoss(nn.Module):
    """
    Kept consistent with original CCDG implementation.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device

    def forward(self, domains_features, domains_labels, temperature=0.7):
        anchor_feature = F.normalize(domains_features, dim=1)
        labels = domains_labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(self.device)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(anchor_feature.shape[0], device=self.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_sum = mask.sum(1)
        zeros_idx = torch.where(mask_sum == 0)[0]
        mask_sum[zeros_idx] = 1

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = (-1.0 * mean_log_prob_pos).mean()
        return loss


class CCDG(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type

        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.use_domain_weight = bool(getattr(configs, "use_domain_weight", False))

        self.domain_contrastive_loss = DomainContrastiveLoss(configs).to(self.device)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

        if self.dataset_type in ("bearing", "uored"):
            self.model = Network_bearing(configs).to(self.device)
        elif self.dataset_type == "fan":
            self.model = Network_fan(configs).to(self.device)
        else:
            raise ValueError("The dataset_type should be bearing, uored or fan!")

        self.optimizer = torch.optim.SGD(list(self.model.parameters()), lr=self.lr)
        self.weight_step = None

    def update(self, minibatches):
        x = torch.cat([batch[0] for batch in minibatches])
        labels = torch.cat([batch[1] for batch in minibatches])

        x = x.to(self.device)
        labels = labels.to(self.device)

        fv, logits = self.model(x)

        if self.weight_step is None:
            self.weight_step = torch.ones(x.shape[0], device=self.device)

        ce_loss = torch.mean(self.cross_entropy_loss(logits, labels) * self.weight_step)
        dc_loss = self.domain_contrastive_loss(fv, labels)

        loss = ce_loss + dc_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss_total": float(loss.detach().cpu().item()),
            "loss_ce": float(ce_loss.detach().cpu().item()),
            "loss_id": float(dc_loss.detach().cpu().item()),
        }

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        loss_acc_result = {
            "loss_total": [],
            "loss_ce": [],
            "loss_id": [],
            "acces": [],
        }

        for step in range(1, self.steps + 1):
            self.train()
            self.current_step = step
            self.logger.info("================Step {}================".format(step))

            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            loss_acc_result["loss_total"].append(losses["loss_total"])
            loss_acc_result["loss_ce"].append(losses["loss_ce"])
            loss_acc_result["loss_id"].append(losses["loss_id"])

            self.logger.info(
                "loss_total_train: \t {loss_total: .4f} \t loss_ce_train: \t {loss_ce: .4f} \t loss_id_train: \t {loss_id: .4f}".format(
                    **losses
                )
            )

            if step % self.checkpoint_freq == 0 or step == self.steps:
                acc_results = self.test_model(test_loaders)
                loss_acc_result["acces"].append(acc_results)

                if self.use_domain_weight:
                    weight_step = torch.from_numpy(1 - np.array(acc_results[1:])).type(torch.float32)
                    self.weight_step = weight_step.repeat((self.batch_size, 1)).T.flatten(0).to(self.device)
                else:
                    self.weight_step = None

                self.logger.info("tgt_train_acc: \t {src_train_acc: .4f}".format(src_train_acc=acc_results[0]))
                for i in range(1, len(acc_results)):
                    self.logger.info("src_train_acc: \t {tgt_train_acc: .4f}".format(tgt_train_acc=acc_results[i]))

        return loss_acc_result

    def test_model(self, loaders, export_pred_paths=None, loader_names=None, split_name=None):
        self.eval()
        acc_results = []

        export_pred_paths = export_pred_paths or [None] * len(loaders)
        loader_names = loader_names or [f"loader_{i}" for i in range(len(loaders))]

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
                        raise RuntimeError(f"Meta batch length mismatch: {len(meta_records)} vs {len(y_pred_np)}")

                    for meta_i, yt, yp in zip(meta_records, y_true_np.tolist(), y_pred_np.tolist()):
                        records.append(
                            {
                                "sample_id": meta_i.get("sample_id"),
                                "file_id": meta_i.get("file_id"),
                                "condition_id": meta_i.get("condition_id"),
                                "y_true": int(yt),
                                "y_pred": int(yp),
                                "method": "CCDG",
                                "run_id": getattr(self.configs, "run_id", ""),
                                "domain_split": split_name or "",
                                "loader_name": loader_names[i],
                            }
                        )

            acc_i, _, _, _ = cal_index(y_true_lst, y_pred_lst)
            acc_results.append(acc_i)

            out_csv = export_pred_paths[i] if i < len(export_pred_paths) else None
            if out_csv and records:
                write_lightweight_preds_csv(records, out_csv)

        self.train()
        return acc_results

    def predict(self, x):
        with torch.no_grad():
            _, logits = self.model(x)
            return torch.max(logits, dim=1)[1]


def build_loaders_for_configs(configs):
    global ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA, ReadMIMII

    if bool(getattr(configs, "use_uored_bridge", False)):
        train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names = build_uored_condition_loaders(
            configs
        )
        configs.datasets_tgt = target_names
        configs.datasets_src = source_names
        return train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names

    if configs.dataset_type == "bearing":
        if any(x is None for x in [ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA]):
            from datasets.load_bearing_data import ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA
        datasets_list = ["CWRU", "UOTTAWA", "MFPT", "DZLRSB"]

    elif configs.dataset_type == "fan":
        if ReadMIMII is None:
            from datasets.load_fan_data import ReadMIMII
        configs.num_classes = 2
        configs.batch_size = 32
        configs.steps = 100

        if configs.fan_section == "sec00":
            datasets_list = ["W", "X", "Y", "Z"]
            section = "00"
        elif configs.fan_section == "sec01":
            datasets_list = ["A", "B", "C"]
            section = "01"
        else:
            datasets_list = ["L1", "L2", "L3", "L4"]
            section = "02"
    else:
        raise ValueError("The dataset_type should be bearing, uored or fan!")

    idx = 0
    dataset_idx = list(range(len(datasets_list)))
    tgt_idx = [idx]
    src_idx = [i for i in dataset_idx if i not in tgt_idx]
    datasets_tgt = [datasets_list[i] for i in tgt_idx]
    datasets_src = [datasets_list[i] for i in src_idx]
    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src

    if configs.dataset_type == "bearing":
        datasets_object_src = [eval("Read" + i + "(configs)") for i in datasets_src]
        datasets_object_tgt = [eval("Read" + i + "(configs)") for i in datasets_tgt]
    else:
        datasets_object_src = [ReadMIMII(i, section=section, configs=configs) for i in datasets_src]
        datasets_object_tgt = [ReadMIMII(i, section=section, configs=configs) for i in datasets_tgt]

    train_test_loaders_src = [dataset.load_dataloaders() for dataset in datasets_object_src]
    train_loaders_src = [train for train, test in train_test_loaders_src]
    test_loaders_src = [test for train, test in train_test_loaders_src]

    train_test_loaders_tgt = [dataset.load_dataloaders() for dataset in datasets_object_tgt]
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt]

    return train_loaders_src, test_loaders_tgt, test_loaders_src, datasets_tgt, datasets_src


def parse_args():
    parser = argparse.ArgumentParser(description="CCDG with optional UORED bridge")
    parser.add_argument("--config", type=str, default=os.path.join(sys.path[0], "config_files", "CCDG_config.yaml"))
    return parser.parse_args()


def main():
    global GenReport

    args = parse_args()
    configs = load_configs(args.config)

    train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names = build_loaders_for_configs(configs)
    train_minibatches_iterator = zip(*train_loaders_src)

    run_root = os.path.join(getattr(configs, "output_root", "Output/CCDG_UORED"), getattr(configs, "run_id", str(time.time())[:10]))
    full_path_log = os.path.join(run_root, "log_files")
    full_path_rep = os.path.join(run_root, "TuneReport")
    pred_dir = os.path.join(run_root, "artifacts")

    os.makedirs(full_path_log, exist_ok=True)
    os.makedirs(full_path_rep, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, "log_file" + currtime))

    model = CCDG(configs)
    for k, v in sorted(vars(configs).items()):
        logger.info("\t{}: {}".format(k, v))

    loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt + test_loaders_src, logger)
    loss_acc_result["loss_total"] = np.array(loss_acc_result["loss_total"])
    loss_acc_result["loss_ce"] = np.array(loss_acc_result["loss_ce"])
    loss_acc_result["loss_id"] = np.array(loss_acc_result["loss_id"])
    loss_acc_result["acces"] = np.array(loss_acc_result["acces"])

    sio.savemat(os.path.join(full_path_log, "loss_acc_result" + currtime + ".mat"), loss_acc_result)

    if bool(getattr(configs, "export_docx_report", False)):
        if GenReport is None:
            from utils.TuneReport import GenReport
        gen_report = GenReport(full_path_rep)
        gen_report.write_file(configs=configs, test_item=None, loss_acc_result=loss_acc_result)
        gen_report.save_file(currtime)

    if bool(getattr(configs, "export_test_preds", True)):
        target_pred_paths = [os.path.join(pred_dir, f"test_preds__{name}.csv") for name in target_names]
        model.test_model(
            test_loaders_tgt,
            export_pred_paths=target_pred_paths,
            loader_names=target_names,
            split_name="target",
        )


if __name__ == "__main__":
    main()