# main.py
import os, time, random, math, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from dataset import load_data, Vocab  # 你已经有的打包函数
from model import EncoderCNN,DecoderLSTM,DecoderTransformer,DecoderBert,DecoderGPT2,EncoderViT

import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval import eval_bleu, show_samples

from logger_utils import init_logger, CSVLogger
from torch.utils.tensorboard import SummaryWriter

target_dir = r"E:\Large_Models"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
import os

# 设置模型缓存目录
os.environ['HF_HOME'] = r'E:\Large_Models'
# main.py 顶部你设置 HF_HOME 的附近加：
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ===== 交叉熵（忽略 <pad>），可选 label smoothing =====
def seq_ce_loss(logits, targets, pad_idx, label_smoothing=0.0):
    # logits: [B, T-1, V], targets: [B, T-1]
    B, Tm1, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B*Tm1, V),
        targets.reshape(B*Tm1),
        ignore_index=pad_idx,
        label_smoothing=label_smoothing
    )

# ===== 单个 epoch 的训练 =====
def train_one_epoch(encoder, decoder, train_loader, optimizer, scaler, device, pad_idx,
                    grad_clip=1.0, label_smoothing=0.0):
    encoder.train(); decoder.train()
    running = 0.0
    pbar = tqdm(train_loader, desc="train", leave=False)
    for images, seqs, lengths in pbar:
        images = images.to(device, non_blocking=True)
        seqs   = seqs.to(device, non_blocking=True)
        targets = seqs[:, 1:]  # 预测下一个词

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            img_emb = encoder(images)          # [B,E] 编码解码
            logits  = decoder(img_emb, seqs)   # [B,T-1,V]
            loss    = seq_ce_loss(logits, targets, pad_idx, label_smoothing)

        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(encoder.parameters()) + list(decoder.parameters()) if p.requires_grad],
                grad_clip
            )
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return running / max(1, len(train_loader))

# ===== 验证（只算 loss） =====
@torch.no_grad()
def evaluate_loss(encoder, decoder, val_loader, device, pad_idx, label_smoothing=0.0):
    encoder.eval(); decoder.eval()
    total = 0.0
    for images, seqs, lengths in tqdm(val_loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        seqs   = seqs.to(device, non_blocking=True)
        targets = seqs[:, 1:]

        img_emb = encoder(images)
        logits  = decoder(img_emb, seqs)
        loss    = seq_ce_loss(logits, targets, pad_idx, label_smoothing)
        total  += loss.item()
    return total / max(1, len(val_loader))

# ===== 训练循环（带保存最优） =====
def fit(encoder, decoder, train_loader, val_loader, optimizer, scaler, device, pad_idx,
        epochs=1, grad_clip=1.0, label_smoothing=0.0, save_path="best.ckpt",
        work_dir="./runs", eval_bleu_every=1, save_every=1, use_tb=True,vocab=None):

    # --- 日志与可视化 ---
    logger = init_logger(work_dir, "train_GPT-2+ViT")
    csvlog = CSVLogger(work_dir, "metrics.csv",
                       fieldnames=("epoch","train_loss","val_loss","val_bleu1","val_bleu4"))
    writer = SummaryWriter(work_dir) if use_tb else None

    best_val = float("inf")
    best_state = None

    global_step = 0
    for ep in range(1, epochs+1):
        # ===== Train =====
        encoder.train(); decoder.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"train epoch {ep}", leave=False)
        for images, seqs, lengths in pbar:
            images = images.to(device, non_blocking=True)
            seqs   = seqs.to(device, non_blocking=True)
            targets = seqs[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                img_emb = encoder(images)
                logits  = decoder(img_emb, seqs)
                loss    = seq_ce_loss(logits, targets, pad_idx, label_smoothing)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in list(encoder.parameters()) + list(decoder.parameters()) if p.requires_grad],
                    grad_clip
                )
            scaler.step(optimizer); scaler.update()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if writer:
                writer.add_scalar("train/step_loss", loss.item(), global_step)

        train_loss = running / max(1, len(train_loader))

        # ===== Validate =====
        val_loss = evaluate_loss(encoder, decoder, val_loader, device, pad_idx, label_smoothing)

        # 可选：每 N 轮评估 BLEU（在 val 上）
        val_bleu1 = val_bleu4 = val_bleu2 = val_bleu3 = None
        if eval_bleu_every and ep % eval_bleu_every == 0:
            metrics = eval_bleu(encoder, decoder, val_loader, vocab=vocab, device=device, max_len=30)

            # 你的 eval_bleu 返回 dict({"BLEU-1":..,"BLEU-2":..,"BLEU-3":..,"BLEU-4":..})
            if isinstance(metrics, dict):
                val_bleu1 = metrics.get("BLEU-1", None)
                val_bleu2 = metrics.get("BLEU-2", None)
                val_bleu3 = metrics.get("BLEU-3", None)
                val_bleu4 = metrics.get("BLEU-4", None)

        # 日志打印
        logger.info(f"[Epoch {ep}] train {train_loss:.4f} | val {val_loss:.4f}"
                    + (f" | BLEU1 {val_bleu1:.4f} | BLEU2 {val_bleu2:.4f}| BLEU3 {val_bleu3:.4f} | BLEU4 {val_bleu4:.4f}" if val_bleu1 is not None else ""))

        # 写 CSV
        csvlog.write(epoch=ep, train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}",
                     val_bleu1=(f"{val_bleu1:.6f}" if val_bleu1 is not None else ""),
                     val_bleu2=(f"{val_bleu2:.6f}" if val_bleu2 is not None else ""),
                     val_bleu3=(f"{val_bleu3:.6f}" if val_bleu3 is not None else ""),
                     val_bleu4=(f"{val_bleu4:.6f}" if val_bleu4 is not None else ""))

        # TensorBoard
        if writer:
            writer.add_scalar("train/epoch_loss", train_loss, ep)
            writer.add_scalar("val/epoch_loss", val_loss, ep)
            if val_bleu1 is not None:
                writer.add_scalar("val/BLEU1", val_bleu1, ep)
                writer.add_scalar("val/BLEU2", val_bleu2, ep)
                writer.add_scalar("val/BLEU3", val_bleu3, ep)
                writer.add_scalar("val/BLEU4", val_bleu4, ep)

        # 保存最优
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": ep,
                "best_val_loss": best_val
            }
            torch.save(best_state, save_path)
            logger.info(f"Saved BEST checkpoint to {save_path} (val_loss={best_val:.4f})")

        # 按周期另存（可选）
        if save_every and ep % save_every == 0:
            ckpt_path = os.path.join(os.path.dirname(save_path), f"epoch_{ep}.ckpt")
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": ep
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    if writer: writer.close()
    csvlog.close()
    return best_state


# ----------------- 0) 基本设置 & 取数据 -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device, flush=True)

base_dir = os.getcwd()
print("base_dir =", base_dir, flush=True)

# 拿到原始 DataLoader（里头是 PIL.Image + ids 的 dataset）
train_loader, val_loader, test_loader, vocab = load_data(base_dir, batch_size=64)
print(f"Vocab size: {len(vocab)}", flush=True)
print("Train/Val/Test samples:",
      len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset), flush=True)


# ===== 2) 初始化模型/优化器/混合精度 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 768
hidden = 512
epochs = 10
lr = 2e-4
grad_clip = 1.0
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", action="store_true",
                    help="指定 -t 时跳过训练，直接加载 best.ckpt 做评估")
args = parser.parse_args()


# encoder = EncoderCNN(embed_dim=embed_dim, train_backbone=False).to(device)
encoder = EncoderViT(embed_dim=embed_dim, train_backbone=False).to(device)

# decoder = DecoderLSTM(len(vocab), embed_dim=embed_dim, hidden=hidden, pad_idx=vocab.pad).to(device)
# decoder = DecoderTransformer(
#     vocab_size=len(vocab),
#     embed_dim=embed_dim,
#     nhead=4,
#     num_layers=8,          # 4 -> 8（或 12）
#     dim_ff=2048,           # 2048 -> 3072（或 4096）
#     dropout=0.1,
#     pad_idx=vocab.pad
# ).to(device)
# decoder = DecoderBert(
#     vocab_size=len(vocab),
#     hidden_size=embed_dim,          # 512，对齐 Encoder 输出维度
#     num_hidden_layers=6,            # 可按显存调 4~8
#     num_attention_heads=8,
#     intermediate_size=2048,
#     pad_idx=vocab.pad, bos_idx=vocab.bos, eos_idx=vocab.eos,
#     dropout=0.1
# ).to(device)
decoder = DecoderGPT2(
    vocab_size=len(vocab),
    embed_dim=embed_dim,
    pad_idx=vocab.pad,  # 确保正确设置 pad_idx
    bos_idx=vocab.bos,  # 设置 BOS token id
    eos_idx=vocab.eos   # 设置 EOS token id
).to(device)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr, weight_decay=1e-4)
# 混合精度使用，减少显存加快速度
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


# with torch.no_grad():
#     decoder.fc.bias.zero_()
#     decoder.fc.bias[vocab.eos] = 2.0
#     # （可选）输出层与嵌入共享权重，通常更稳
#     decoder.fc.weight = decoder.embed.weight

# 开训！
# ===== 加载已有的 best.ckpt =====
if __name__ == "__main__":
    if not args.test:   # 默认训练
        fit(encoder, decoder, train_loader, val_loader,
            optimizer, scaler, device, pad_idx=vocab.pad,
            epochs=epochs, grad_clip=grad_clip, label_smoothing=0.1,
            save_path=os.path.join(base_dir,'best.ckpt'),
            work_dir="./runs", eval_bleu_every=1, save_every=1, use_tb=True, vocab=vocab)
    else:               # 指定 -t 就 eval
        ckpt_path = os.path.join(base_dir, 'best.ckpt')
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        print(f"Loaded checkpoint from {ckpt_path}")
_ = eval_bleu(encoder, decoder, test_loader, vocab, device, max_len=30)
show_samples(encoder, decoder, test_loader, vocab, device, n=5, max_len=20)


