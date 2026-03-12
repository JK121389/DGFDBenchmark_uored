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
    else:
        configs.device = "cpu"
    return configs


def _safe_to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_embedding_npz(save_path, payload: dict):
    """
    统一保存 npz，字符串字段转 object array，数值字段转 numpy array。
    """
    save_dict = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, list):
            if len(v) == 0:
                save_dict[k] = np.array([], dtype=object)
            else:
                # 纯字符串/混合元信息统一用 object，更稳
                if isinstance(v[0], str) or v[0] is None:
                    save_dict[k] = np.array(v, dtype=object)
                else:
                    save_dict[k] = np.asarray(v)
        else:
            save_dict[k] = _safe_to_numpy(v)
    np.savez_compressed(save_path, **save_dict)


class CausalLoss(nn.Module):
    """
    Consistent with original WhiteningNet implementation.
    """
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes

    def forward(self, flatten_features, data_label):
        list_category = [[] for _ in range(self.num_classes)]
        for i, fv in zip(data_label, flatten_features):
            fv = torch.reshape(fv, (1, fv.size(0)))
            list_category[int(i)].append(fv)

        total_causal_loss = 0.0
        for i in range(self.num_classes):
            if len(list_category[i]) == 0:
                continue
            fm_i = torch.cat(tuple(list_category[i]), dim=0)
            fm_i_mean = torch.mean(fm_i, dim=0, keepdim=True)
            causal_i = torch.sum(torch.mean((fm_i - fm_i_mean).pow(2), dim=0))
            total_causal_loss = total_causal_loss + causal_i

        return total_causal_loss


class WhitenNet(nn.Module):
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
        self.use_learning_rate_sheduler = bool(getattr(configs, "use_learning_rate_sheduler", False))

        if self.dataset_type in ("bearing", "uored"):
            self.model = Network_bearing(configs).to(self.device)
        elif self.dataset_type == "fan":
            self.model = Network_fan(configs).to(self.device)
        else:
            raise ValueError("The dataset_type should be bearing, uored or fan!")

        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)
        self.cl_loss = nn.CrossEntropyLoss(reduction="none")
        self.causal_loss = CausalLoss(configs)

        self.lamda_causal = configs.lamda_causal
        self.num_domains = len(configs.datasets_src)
        self.weight_step = None

    def adjust_learning_rate(self, step):
        lr = self.lr
        if self.configs.cos:
            lr *= 0.5 * (1.0 + math.cos(math.pi * step / self.steps))
        else:
            for milestone in self.configs.schedule:
                lr *= 0.5 if step >= milestone else 1.0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])

        x = x.to(self.device)
        labels = labels.to(self.device)

        feature_vectors, logits = self.model(x)

        if self.use_domain_weight:
            if self.weight_step is None:
                self.weight_step = torch.ones(x.shape[0], device=self.device)
            else:
                ce_values = self.cl_loss(logits, labels)
                ce_values_2d = torch.reshape(ce_values, (self.num_domains, self.batch_size))
                ce_value_domain = torch.mean(ce_values_2d, dim=1)
                ce_value_sum = torch.sum(ce_value_domain)
                weight_step = 1 + ce_value_domain / ce_value_sum
                self.weight_step = weight_step.repeat((self.batch_size, 1)).T.flatten(0).to(self.device)
        else:
            self.weight_step = torch.ones(x.shape[0], device=self.device)

        cl_loss = torch.mean(self.cl_loss(logits, labels) * self.weight_step)
        causal_loss = self.causal_loss(feature_vectors, labels)
        total_loss = cl_loss + self.lamda_causal * causal_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "cl": float(cl_loss.detach().cpu().item()),
            "causal": float(causal_loss.detach().cpu().item()),
            "total": float(total_loss.detach().cpu().item()),
        }

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        loss_acc_result = {
            "loss_total": [],
            "loss_causal": [],
            "loss_cl": [],
            "acces": [],
        }

        for step in range(1, self.steps + 1):
            self.train()
            self.current_step = step
            self.logger.info("================Step {}================".format(step))

            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            loss_acc_result["loss_total"].append(losses["total"])
            loss_acc_result["loss_cl"].append(losses["cl"])
            loss_acc_result["loss_causal"].append(losses["causal"])

            self.logger.info(
                "loss_total_train: \t {total: .4f} \t loss_cl_train: \t {cl: .4f} \t loss_causal_train: \t {causal: .4f}".format(
                    **losses
                )
            )

            if step % self.checkpoint_freq == 0 or step == self.steps:
                acc_results = self.test_model(test_loaders)
                loss_acc_result["acces"].append(acc_results)

                self.logger.info("tgt_train_acc: \t {src_train_acc: .4f}".format(src_train_acc=acc_results[0]))
                for i in range(1, len(acc_results)):
                    self.logger.info("src_train_acc: \t {tgt_train_acc: .4f}".format(tgt_train_acc=acc_results[i]))

        return loss_acc_result

    def test_model(
        self,
        loaders,
        export_pred_paths=None,
        loader_names=None,
        split_name=None,
        export_embed_paths=None,
        export_embed_merged_path=None,
    ):
        """
        保持原有测试/预测逻辑不变，只新增可选 embeddings 导出：
        - export_embed_paths: 每个 loader 一个 npz
        - export_embed_merged_path: 所有 loader 合并一个 npz
        """
        self.eval()
        acc_results = []

        export_pred_paths = export_pred_paths or [None] * len(loaders)
        export_embed_paths = export_embed_paths or [None] * len(loaders)
        loader_names = loader_names or [f"loader_{i}" for i in range(len(loaders))]

        merged_features = []
        merged_logits = []
        merged_y_true = []
        merged_y_pred = []
        merged_sample_id = []
        merged_file_id = []
        merged_condition_id = []
        merged_loader_name = []
        merged_method = []
        merged_run_id = []
        merged_domain_split = []

        for i, the_loader in enumerate(loaders):
            y_pred_lst = []
            y_true_lst = []
            records = []

            # per-loader embeddings cache
            feat_list = []
            logits_list = []
            yt_list = []
            yp_list = []
            sample_id_list = []
            file_id_list = []
            condition_id_list = []
            loader_name_list = []
            method_list = []
            run_id_list = []
            domain_split_list = []

            for batched_data in the_loader:
                if isinstance(batched_data, (list, tuple)) and len(batched_data) == 3:
                    x, label_fault, meta = batched_data
                else:
                    x, label_fault = batched_data
                    meta = None

                x = x.to(self.device)
                label_fault = label_fault.to(self.device)

                with torch.no_grad():
                    feature_vectors, logits = self.model(x)
                    label_pred = torch.max(logits, dim=1)[1]

                y_pred_np = label_pred.detach().cpu().numpy()
                y_true_np = label_fault.detach().cpu().numpy()

                y_pred_lst.extend(y_pred_np.tolist())
                y_true_lst.extend(y_true_np.tolist())

                # embeddings 缓存
                feat_np = feature_vectors.detach().cpu().numpy()
                logits_np = logits.detach().cpu().numpy()

                feat_list.append(feat_np)
                logits_list.append(logits_np)
                yt_list.extend(y_true_np.tolist())
                yp_list.extend(y_pred_np.tolist())

                if meta is not None:
                    meta_records = collect_batch_meta(meta)
                    if len(meta_records) != len(y_pred_np):
                        raise RuntimeError(f"Meta batch length mismatch: {len(meta_records)} vs {len(y_pred_np)}")

                    for meta_i, yt, yp in zip(meta_records, y_true_np.tolist(), y_pred_np.tolist()):
                        sample_id = meta_i.get("sample_id")
                        file_id = meta_i.get("file_id")
                        condition_id = meta_i.get("condition_id")

                        records.append(
                            {
                                "sample_id": sample_id,
                                "file_id": file_id,
                                "condition_id": condition_id,
                                "y_true": int(yt),
                                "y_pred": int(yp),
                                "method": "WhiteningNet",
                                "run_id": getattr(self.configs, "run_id", ""),
                                "domain_split": split_name or "",
                                "loader_name": loader_names[i],
                            }
                        )

                        sample_id_list.append(sample_id)
                        file_id_list.append(file_id)
                        condition_id_list.append(condition_id)
                        loader_name_list.append(loader_names[i])
                        method_list.append("WhiteningNet")
                        run_id_list.append(getattr(self.configs, "run_id", ""))
                        domain_split_list.append(split_name or "")
                else:
                    # 没有 meta 时也保证 embeddings 可导出
                    batch_n = len(y_pred_np)
                    sample_id_list.extend([None] * batch_n)
                    file_id_list.extend([None] * batch_n)
                    condition_id_list.extend([loader_names[i]] * batch_n)
                    loader_name_list.extend([loader_names[i]] * batch_n)
                    method_list.extend(["WhiteningNet"] * batch_n)
                    run_id_list.extend([getattr(self.configs, "run_id", "")] * batch_n)
                    domain_split_list.extend([split_name or ""] * batch_n)

            acc_i, _, _, _ = cal_index(y_true_lst, y_pred_lst)
            acc_results.append(acc_i)

            out_csv = export_pred_paths[i] if i < len(export_pred_paths) else None
            if out_csv and records:
                write_lightweight_preds_csv(records, out_csv)

            # 保存当前 loader 的 embeddings
            out_embed = export_embed_paths[i] if i < len(export_embed_paths) else None
            if out_embed is not None:
                os.makedirs(os.path.dirname(out_embed), exist_ok=True)

                payload = {
                    "features": np.concatenate(feat_list, axis=0) if len(feat_list) > 0 else np.empty((0, 0), dtype=np.float32),
                    "logits": np.concatenate(logits_list, axis=0) if len(logits_list) > 0 else np.empty((0, 0), dtype=np.float32),
                    "y_true": np.asarray(yt_list, dtype=np.int64),
                    "y_pred": np.asarray(yp_list, dtype=np.int64),
                    "sample_id": sample_id_list,
                    "file_id": file_id_list,
                    "condition_id": condition_id_list,
                    "loader_name": loader_name_list,
                    "method": method_list,
                    "run_id": run_id_list,
                    "domain_split": domain_split_list,
                }
                _save_embedding_npz(out_embed, payload)

            # 汇总到 merged
            if len(feat_list) > 0:
                merged_features.append(np.concatenate(feat_list, axis=0))
                merged_logits.append(np.concatenate(logits_list, axis=0))
                merged_y_true.extend(yt_list)
                merged_y_pred.extend(yp_list)
                merged_sample_id.extend(sample_id_list)
                merged_file_id.extend(file_id_list)
                merged_condition_id.extend(condition_id_list)
                merged_loader_name.extend(loader_name_list)
                merged_method.extend(method_list)
                merged_run_id.extend(run_id_list)
                merged_domain_split.extend(domain_split_list)

        # 保存 merged embeddings
        if export_embed_merged_path is not None:
            os.makedirs(os.path.dirname(export_embed_merged_path), exist_ok=True)
            merged_payload = {
                "features": np.concatenate(merged_features, axis=0) if len(merged_features) > 0 else np.empty((0, 0), dtype=np.float32),
                "logits": np.concatenate(merged_logits, axis=0) if len(merged_logits) > 0 else np.empty((0, 0), dtype=np.float32),
                "y_true": np.asarray(merged_y_true, dtype=np.int64),
                "y_pred": np.asarray(merged_y_pred, dtype=np.int64),
                "sample_id": merged_sample_id,
                "file_id": merged_file_id,
                "condition_id": merged_condition_id,
                "loader_name": merged_loader_name,
                "method": merged_method,
                "run_id": merged_run_id,
                "domain_split": merged_domain_split,
            }
            _save_embedding_npz(export_embed_merged_path, merged_payload)

        self.train()
        return acc_results

    def predict(self, x):
        with torch.no_grad():
            _, logits = self.model(x)
            return torch.max(logits, dim=1)[1]


def build_loaders_for_configs(configs):
    global ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA, ReadMIMII

    if bool(getattr(configs, "use_uored_bridge", False)):
        train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names = build_uored_condition_loaders(configs)
        configs.datasets_tgt = target_names
        configs.datasets_src = source_names
        return train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names

    if configs.dataset_type == "bearing":
        configs.fan_section = None
        configs.num_classes = 3
        configs.batch_size = 64
        configs.steps = 200

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
    parser = argparse.ArgumentParser(description="WhiteningNet with optional UORED bridge")
    parser.add_argument("--config", type=str, default=os.path.join(sys.path[0], "config_files", "WhiteningNet_config.yaml"))
    return parser.parse_args()


def main():
    global GenReport

    args = parse_args()
    configs = load_configs(args.config)

    train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names = build_loaders_for_configs(configs)
    train_minibatches_iterator = zip(*train_loaders_src)

    # 仅修改输出目录组织方式，不影响原模型训练/测试逻辑
    base_run_id = getattr(configs, "run_id", str(time.time())[:10])
    timestamp_tag = time.strftime("%Y%m%d_%H%M%S")
    run_id_effective = f"{base_run_id}__embed_{timestamp_tag}"
    configs.run_id = run_id_effective

    run_root = os.path.join(
        getattr(configs, "output_root", "Output/WhiteningNet_UORED"),
        run_id_effective,
    )
    full_path_log = os.path.join(run_root, "log_files")
    full_path_rep = os.path.join(run_root, "TuneReport")
    pred_dir = os.path.join(run_root, "artifacts")

    os.makedirs(full_path_log, exist_ok=True)
    os.makedirs(full_path_rep, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    currtime = str(time.time())[:10]
    logger = create_logger(os.path.join(full_path_log, "log_file" + currtime))

    model = WhitenNet(configs)
    for k, v in sorted(vars(configs).items()):
        logger.info("\t{}: {}".format(k, v))

    loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt + test_loaders_src, logger)
    loss_acc_result["loss_total"] = np.array(loss_acc_result["loss_total"])
    loss_acc_result["loss_cl"] = np.array(loss_acc_result["loss_cl"])
    loss_acc_result["loss_causal"] = np.array(loss_acc_result["loss_causal"])
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
    else:
        target_pred_paths = [None] * len(target_names)

    # 新增：target test embeddings 导出
    export_test_embeddings = bool(getattr(configs, "export_test_embeddings", True))
    if export_test_embeddings:
        target_embed_paths = [os.path.join(pred_dir, f"test_embeddings__{name}.npz") for name in target_names]
        merged_embed_path = os.path.join(pred_dir, "test_embeddings.npz")
    else:
        target_embed_paths = [None] * len(target_names)
        merged_embed_path = None

    model.test_model(
        test_loaders_tgt,
        export_pred_paths=target_pred_paths,
        loader_names=target_names,
        split_name="target",
        export_embed_paths=target_embed_paths,
        export_embed_merged_path=merged_embed_path,
    )

    logger.info("Run finished.")
    logger.info(f"Artifacts saved under: {run_root}")


if __name__ == "__main__":
    main()