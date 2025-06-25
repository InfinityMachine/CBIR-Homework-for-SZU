"""image_retrieval_system_cpu.py
============================================================
CPU 图像检索（Holidays TXT & Oxford‑5k）
============================================================

安装依赖（conda 示例）
------------------------------------------------------------
```bash
conda create -n ir python=3.10 -y
conda activate ir
conda install -c conda-forge faiss-cpu
pip install torch torchvision pillow tqdm
```
"""

from __future__ import annotations

# ------------------------------------
# 模块导入：支持命令行解析、数据处理、图像加载与深度学习模型构建
# ------------------------------------
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss  # Facebook AI Similarity Search 库，用于高效向量检索
import numpy as np  # 数值计算库
import torch  # 深度学习框架
import torch.nn as nn  # 神经网络模块
import torchvision.transforms as T  # 图像预处理模块
from PIL import Image  # 图像读写与处理


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
# 配置类：集中管理数据集路径、模型架构及超参数设置
@dataclass
class Config:
    # 数据集根路径
    holidays_root: str = "Holidays"
    oxford_root: str = "Oxford5k"
    # CNN模型名称与特征维度
    cnn_arch: str = "resnet50"
    feat_dim: int = 2048
    # 批处理大小与输入图像短边尺寸
    batch_size: int = 32
    img_size: int = 224

    # 存储路径
    index_path: str = "faiss.index"
    feats_path: str = "features.npy"
    paths_path: str = "img_paths.json"


cfg = Config()  # 实例化参数配置对象

# ------------------------------------
# 图像预处理流水线：调整尺寸、中心裁剪、Tensor转换与标准化
# ------------------------------------
_preprocess = T.Compose(
    [
        T.Resize(cfg.img_size),
        T.CenterCrop(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ------------------------------------
# 特征提取器 FeatureExtractor：
# • 加载预训练 ResNet-50
# • 去除分类层，仅保留卷积特征
# • 输出 L2 归一化的全局特征向量
# ------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, arch: str = "resnet50") -> None:
        super().__init__()
        # 从torchvision hub加载预训练模型
        model = torch.hub.load("pytorch/vision", arch, pretrained=True)
        # 去除最后全连接层，仅保留特征提取部分
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.backbone.eval()  # 模型推理模式，关闭 Dropout/BatchNorm 更新

    @torch.inference_mode()
    def forward(self, img: Image.Image | torch.Tensor) -> torch.Tensor:
        # 如果输入是 PIL Image，先进行预处理
        if isinstance(img, Image.Image):
            img = _preprocess(img)
        # 添加批次维度，变为1×3×H×W
        img = img.unsqueeze(0)
        # 提取特征，去除多余维度后得到向量
        feat = self.backbone(img).squeeze()
        # 对特征向量进行 L2 归一化
        return nn.functional.normalize(feat, p=2, dim=0)


# ------------------------------------
# IO 函数 load_image：加载图像并统一为 RGB 模式，确保后续一致性
# ------------------------------------
def load_image(p: str | Path) -> Image.Image:
    # 从文件或路径加载图像并转换为RGB格式
    return Image.open(p).convert("RGB")


# ---------------------------------------------------------------------------
# 索引构建
# ---------------------------------------------------------------------------
# 索引构建器：遍历图像目录，批量提取特征，并构建和保存FAISS索引
class Indexer:
    def __init__(self, extractor: FeatureExtractor, batch: int):
        self.ext, self.batch = extractor, batch

    @staticmethod
    def _images(root: Path):
        # 返回目录下所有支持格式的图像路径
        return sorted(
            p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}
        )

    def build(self, roots: List[str]):
        # 收集所有图像文件路径
        imgs: List[Path] = []
        for r in roots:
            imgs.extend(self._images(Path(r)))
        n = len(imgs)
        # 初始化特征矩阵（N×D）
        feats = np.zeros((n, cfg.feat_dim), dtype="float32")
        # 分批次提取特征
        for i in range(0, n, self.batch):
            batch = imgs[i : i + self.batch]
            tensors = torch.stack([_preprocess(load_image(p)) for p in batch])
            with torch.inference_mode():
                vec = self.ext.backbone(tensors).squeeze(-1).squeeze(-1)
                vec = nn.functional.normalize(vec, p=2, dim=1)
            feats[i : i + len(batch)] = vec.cpu().numpy()
        # 保存特征及路径到文件
        np.save(cfg.feats_path, feats)
        with open(cfg.paths_path, "w") as f:
            json.dump([str(p) for p in imgs], f)
        # 构建FAISS索引并写入磁盘
        index = faiss.IndexFlatIP(cfg.feat_dim)
        index.add(feats)
        faiss.write_index(index, cfg.index_path)
        print(f"[Indexer] 构建完毕，共处理 {n} 张图像。索引已保存至 {cfg.index_path}")


# ---------------------------------------------------------------------------
# 检索与评估
# ---------------------------------------------------------------------------
# 检索系统：加载索引与特征数据，提供搜索与评估接口
class RetrievalSystem:
    def __init__(self):
        # 读取已有的FAISS索引和图像路径列表
        self.index = faiss.read_index(cfg.index_path)
        self.img_paths = json.load(open(cfg.paths_path))
        self.ext = FeatureExtractor(cfg.cnn_arch)
        self._fname = [Path(p).name for p in self.img_paths]

    def _encode(self, img: str | Path | Image.Image) -> np.ndarray:
        # 将图像编码为模型特征向量并转换为 NumPy 数组
        if not isinstance(img, Image.Image):
            img = load_image(img)
        vec = self.ext(img).numpy()[None, :].astype("float32")
        return vec

    def search(self, img: str | Path, top: int) -> List[Tuple[str, float]]:
        # 对编码向量执行 FAISS 检索，返回路径与相似度分数
        scores, idx = self.index.search(self._encode(img), top)
        return [(self.img_paths[i], float(scores[0, j])) for j, i in enumerate(idx[0])]

    @staticmethod
    def _ap(ranks: List[int], nrel: int) -> float:
        # 依据相关项位置计算平均精度（AP）
        if nrel == 0:
            return 0.0
        hits, score = 0, 0.0
        for r in sorted(ranks):
            hits += 1
            score += hits / (r + 1)  # 使用原始排名(r+1)计算精度
        return score / nrel

    # ------------- Holidays TXT -----------------
    def eval_holidays(self) -> None:
        # 评估Holidays数据集的mAP
        lst = Path(cfg.holidays_root) / "lists.txt"
        if not lst.exists():
            print("[Eval] Holidays lists.txt 文件缺失")
            return
        rel, queries = {}, []
        seen = set()
        # 解析lists.txt，构建查询列表和相关集
        for line in lst.read_text().splitlines():
            if not line.strip():
                continue
            img, gid = line.split()[:2]
            p = str(Path(cfg.holidays_root) / "jpg" / img)
            rel.setdefault(gid, []).append(p)
            if gid not in seen:
                queries.append((p, gid))
                seen.add(gid)
        aps = []
        # 对每个查询计算AP，并输出mAP
        for q, gid in queries:
            relevant = set(rel[gid])
            relevant.discard(q)
            ranks = [
                i
                for i, (pp, _) in enumerate(self.search(q, len(self.img_paths)))
                if pp in relevant
            ]
            aps.append(self._ap(ranks, len(relevant)))
        print(f"[Eval-Holidays] mAP = {np.mean(aps):.4f}")

    # ------------- Oxford-5k --------------------
    def eval_oxford(self) -> None:
        # 评估Oxford-5k数据集的mAP
        gt_dir = Path(cfg.oxford_root) / "gt"
        if not gt_dir.exists():
            print("[Eval] Oxford 地面真值文件夹缺失")
            return
        aps = []
        for qf in gt_dir.glob("*_query.txt"):
            tag = qf.stem[:-6]
            # 解析查询图像名称
            name = qf.read_text().split()[0]
            if name.startswith("oxc") and "_" in name:
                name = name.split("_", 1)[1]
            qimg = Path(cfg.oxford_root) / "jpg" / f"{name}.jpg"

            # 读取good、ok和junk集合
            def read_set(kind):
                f = gt_dir / f"{tag}_{kind}.txt"
                return (
                    set(x.strip() + ".jpg" for x in f.read_text().splitlines())
                    if f.exists()
                    else set()
                )

            good, ok = read_set("good"), read_set("ok")
            junk = read_set("junk")
            rel = good | ok
            if not rel:
                continue
            # 执行检索并过滤junk，收集相关项的排名
            scores, idx = self.index.search(self._encode(qimg), len(self.img_paths))
            ranks, k = [], 0
            for i in idx[0].tolist():
                fname = self._fname[i]
                if fname in junk:
                    continue
                if fname in rel:
                    ranks.append(k)
                k += 1
            aps.append(self._ap(ranks, len(rel)))
        print(
            f"[Eval-Oxford5k] mAP = {np.mean(aps):.4f}"
            if aps
            else "[Eval-Oxford5k] mAP = nan"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# 命令行接口：解析子命令并调用对应功能（构建索引、评估、查询）
def main() -> None:
    pa = argparse.ArgumentParser("CPU 图像检索系统")
    sub = pa.add_subparsers(dest="cmd", required=True)
    # 构建索引子命令
    b = sub.add_parser("build-index")
    b.add_argument("--holidays-root", default=cfg.holidays_root)
    b.add_argument("--oxford-root", default=cfg.oxford_root)
    # 评估子命令
    sub.add_parser("evaluate")
    # 查询子命令
    q = sub.add_parser("query")
    q.add_argument("--img", required=True, help="查询图像路径")
    q.add_argument("--top", type=int, default=5, help="返回Top-K结果数")

    args = pa.parse_args()
    # 覆盖默认路径配置
    cfg.holidays_root = getattr(args, "holidays_root", cfg.holidays_root)
    cfg.oxford_root = getattr(args, "oxford_root", cfg.oxford_root)
    # 根据子命令调用功能
    if args.cmd == "build-index":
        Indexer(FeatureExtractor(cfg.cnn_arch), cfg.batch_size).build(
            [str(Path(cfg.holidays_root) / "jpg"), str(Path(cfg.oxford_root) / "jpg")]
        )
    elif args.cmd == "evaluate":
        sys = RetrievalSystem()
        sys.eval_holidays()
        sys.eval_oxford()
    elif args.cmd == "query":
        sys = RetrievalSystem()
        for path, score in sys.search(args.img, args.top):
            print(f"{score:.4f}\t{path}")


if __name__ == "__main__":
    main()
