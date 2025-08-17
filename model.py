import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
class EncoderCNN(nn.Module):
    def __init__(self,embed_dim =512,train_backbone=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) #取出除了最后一层分类层 因为是分类用的
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone #先冻结
        self.proj = nn.Linear(2048,embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self,images):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            feats = self.backbone(images).squeeze(-1).squeeze(-1)  # [B,2048]
        emb = self.bn(self.proj(feats))  # [B,E]
        return emb
class DecoderLSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim=512,hidden=512,num_layers=1,pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
        self.init_h = nn.Linear(embed_dim, hidden)
        self.init_c = nn.Linear(embed_dim, hidden)
        self.pad_idx = pad_idx
    def forward(self,img_emb,seq): #预训练阶段 学习调参
        B, T = seq.size() #batch_size token_length
        h0 = torch.tanh(self.init_h(img_emb)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(img_emb)).unsqueeze(0)
        x = self.embed(seq[:,:-1]) # [B,T-1,E]
        out , _ = self.lstm(x,(h0,c0))
        logits = self.fc(out)
        return logits
    # 一步一步解码，bos是开始 eos是结束
    @torch.no_grad()
    def greedy_decode(self, img_emb, bos_idx, eos_idx, max_len=30): #验证阶段
        B = img_emb.size(0)
        h = torch.tanh(self.init_h(img_emb)).unsqueeze(0)
        c = torch.tanh(self.init_c(img_emb)).unsqueeze(0)
        # 创建一个batch,1的 每个句子开始都是bos
        cur = torch.full((B, 1), bos_idx, dtype=torch.long, device=img_emb.device)
        outs = []
        # 最多生成30长度，如果超过直接break
        for _ in range(max_len):
            emb = self.embed(cur[:, -1:])  # [B,1,E]
            out, (h, c) = self.lstm(emb, (h, c))  # [B,1,H]
            logits = self.fc(out.squeeze(1))  # [B,V]
            nxt = logits.argmax(-1, keepdim=True)  # [B,1]
            outs.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
            if (nxt.squeeze(1) == eos_idx).all():
                break
        return torch.cat(outs, dim=1)  # [B,L]


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """标准正弦位置编码（可替换为 nn.Embedding 做可学习位置编码）"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不训练

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1,T,D] 广播


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, nhead=8, num_layers=4,
                 dim_ff=2048, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = PositionalEncoding(embed_dim, max_len=512)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True  # 重要：用 [B,T,D]
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # 把图像向量当作 memory（长度=1 的“图像 token”）
        self.mem_proj = nn.Linear(embed_dim, embed_dim)

        self.fc = nn.Linear(embed_dim, vocab_size)

        # （可选）权重共享
        self.fc.weight = self.embed.weight

    def _generate_square_subsequent_mask(self, sz: int, device):
        # [T,T]，下三角为 False（可看），上三角为 True（mask 掉）
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, img_emb, seq):
        """
        训练阶段（teacher forcing）：
        img_emb: [B, E]，由 EncoderCNN 输出
        seq:     [B, T]，含 <bos> ... <eos>
        返回 logits: [B, T-1, V]
        """
        B, T = seq.size()
        tgt_in = seq[:, :-1]  # 预测下一个词
        tgt_emb = self.embed(tgt_in)  # [B, T-1, E]
        tgt_emb = self.pos(tgt_emb)  # 加位置

        # causal mask（注意 True 代表“不可见”）
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(1), seq.device)  # [T-1,T-1]

        # padding mask：True 表示 padding 位置需要被 mask
        tgt_key_padding_mask = (tgt_in == self.pad_idx)  # [B,T-1]

        # memory：把图像 embedding 投影后作为长度=1 的序列
        memory = self.mem_proj(img_emb).unsqueeze(1)  # [B,1,E]

        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None
        )  # [B,T-1,E]
        logits = self.fc(dec_out)  # [B,T-1,V]
        return logits

    @torch.no_grad()
    def greedy_decode(self, img_emb, bos_idx, eos_idx, max_len=30):
        """
        推理阶段：从 <bos> 开始自回归生成
        返回 ids: [B, Lgen]
        """
        device = img_emb.device
        B = img_emb.size(0)
        memory = self.mem_proj(img_emb).unsqueeze(1)  # [B,1,E]

        cur = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)  # [B,1]
        outs = []

        for _ in range(max_len):
            tgt_emb = self.embed(cur)  # [B,t,E]
            tgt_emb = self.pos(tgt_emb)
            t = tgt_emb.size(1)
            tgt_mask = self._generate_square_subsequent_mask(t, device)  # [t,t]

            dec_out = self.decoder(
                tgt=tgt_emb, memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(cur == self.pad_idx)
            )  # [B,t,E]
            logits = self.fc(dec_out[:, -1, :])  # 只取最后一步 [B,V]
            nxt = logits.argmax(-1, keepdim=True)  # [B,1]
            outs.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
            if (nxt.squeeze(1) == eos_idx).all():
                break

        return torch.cat(outs, dim=1)  # [B,L]

# decoder_bert.py
import torch
import torch.nn as nn
from transformers import BertConfig, BertLMHeadModel


class DecoderBert(nn.Module):
    """
    用 BERT 做自回归解码器（带 cross-attention）：
    - is_decoder=True  → 自回归因果 mask
    - add_cross_attention=True → 与图像 memory 交叉注意力
    - vocab_size 用你的自定义 vocab（不是 Bert 自带词表）
    """
    def __init__(self, vocab_size, hidden_size=512, num_hidden_layers=6,
                 num_attention_heads=8, intermediate_size=2048,
                 pad_idx=0, bos_idx=1, eos_idx=2, dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            is_decoder=True,
            add_cross_attention=True,
            pad_token_id=pad_idx
        )
        # 带 LM 头的 BERT（输出 logits）
        self.bert = BertLMHeadModel(config)

        # 把图像向量映射到 BERT hidden 维度；如果 EncoderCNN 输出维度 != hidden_size，改 in_features
        self.img_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, img_emb, seq):
        """
        训练 (teacher forcing)
        img_emb: [B,E]  （E ≈ hidden_size）
        seq    : [B,T]  含 <bos> ... <eos>
        return : logits [B,T-1,V]
        """
        input_ids = seq[:, :-1]                              # 预测下一个词
        attention_mask = (input_ids != self.pad_idx).long()  # [B,T-1]
        memory = self.img_proj(img_emb).unsqueeze(1)         # [B,1,H]
        memory_mask = torch.ones(memory.size()[:-1], dtype=torch.long, device=memory.device)  # [B,1]

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask
        )
        logits = out.logits   # [B,T-1,V]
        return logits

    @torch.no_grad()
    def greedy_decode(self, img_emb, bos_idx, eos_idx, max_len=30):
        """
        自回归生成；返回 [B, Lgen]
        """
        device = img_emb.device
        B = img_emb.size(0)
        memory = self.img_proj(img_emb).unsqueeze(1)          # [B,1,H]
        memory_mask = torch.ones(memory.size()[:-1], dtype=torch.long, device=device)
        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        outs = []
        for _ in range(max_len):
            attn_mask = (ys != self.pad_idx).long()
            out = self.bert(
                input_ids=ys,
                attention_mask=attn_mask,
                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask
            )
            logits = out.logits[:, -1, :]                     # [B,V]
            nxt = logits.argmax(-1, keepdim=True)             # [B,1]
            outs.append(nxt)
            ys = torch.cat([ys, nxt], dim=1)
            if (nxt.squeeze(1) == eos_idx).all():
                break
        return torch.cat(outs, dim=1)


import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig

from transformers import ViTModel

class EncoderViT(nn.Module):
    def __init__(self, embed_dim=512, train_backbone=False,
                 vit_name_or_path='google/vit-base-patch16-224-in21k',
                 use_timm_fallback=True):
        super().__init__()
        self.out_dim = embed_dim

        backbone = None
        # 优先：Transformers 离线加载
        try:
            from transformers import ViTModel
            import os
            # 强制离线模式，防止尝试联网
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            backbone = ViTModel.from_pretrained(
                vit_name_or_path,
                local_files_only=True  # 只用本地缓存
            )
            print("[ViT] Loaded from local cache:", vit_name_or_path, flush=True)
        except Exception as e:
            print("[ViT] Local pretrained not available ->", e, flush=True)
            if use_timm_fallback:
                # 其次：timm 随机初始化版（不下载权重）
                try:
                    import timm
                    backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
                    self.is_timm = True
                    print("[ViT] Using timm vit_base_patch16_224 (random init).", flush=True)
                except Exception as e2:
                    print("[ViT] timm fallback failed ->", e2, flush=True)

        if backbone is None:
            # 最后：Transformers 随机初始化版
            from transformers import ViTConfig, ViTModel
            cfg = ViTConfig()
            backbone = ViTModel(cfg)
            print("[ViT] Using transformers ViTModel with random init.", flush=True)

        self.is_timm = hasattr(backbone, 'forward_features')  # timm 分支判断
        self.backbone = backbone

        # 冻结与否
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone

        # 输出维度（transformers/timm 的 hidden_size 不同对象名）
        hidden = (getattr(getattr(self.backbone, "config", None), "hidden_size", None)
                  or getattr(self.backbone, "num_features", None) or 768)

        self.proj = nn.Linear(hidden, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, images):
        if self.is_timm:
            feats = self.backbone.forward_features(images)  # [B, tokens, C]
            if feats.dim() == 3:
                feats = feats[:, 0, :]  # 取 CLS
        else:
            feats = self.backbone(images).last_hidden_state[:, 0, :]  # transformers CLS

        emb = self.bn(self.proj(feats))  # [B, E]
        return emb

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from transformers import GPT2Config, GPT2LMHeadModel

class DecoderGPT2(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, pad_idx=0, bos_idx=1, eos_idx=2):
        super().__init__()

        # 加载 GPT2 配置，启用 cross-attention
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        gpt2_config.add_cross_attention = True  # 启用 cross-attention

        # 使用 GPT2 模型并配置其为解码器
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)

        # 将图像嵌入映射到 GPT2 的输入维度
        self.img_proj = nn.Linear(embed_dim, self.gpt2.config.n_embd)

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def forward(self, img_emb, seq):
        """
        训练阶段：输入图像嵌入和目标序列
        img_emb: [B, E]  （图像的嵌入向量）
        seq:     [B, T]  含 <bos> ... <eos>
        返回 logits: [B, T-1, V]
        """
        B, T = seq.size()
        input_ids = seq[:, :-1]  # 预测下一个词
        attention_mask = (input_ids != self.pad_idx).long()  # [B,T-1]

        # 将图像嵌入映射为 GPT2 的输入维度
        memory = self.img_proj(img_emb).unsqueeze(1)  # [B, 1, E] -> [B, 1, n_embd]

        # 将[CLS] token对应的图像embedding作为 memory 传入GPT2
        output = self.gpt2(input_ids=input_ids,
                           attention_mask=attention_mask,
                           encoder_hidden_states=memory)

        logits = output.logits  # [B, T-1, V]
        return logits

    @torch.no_grad()
    def greedy_decode(self, img_emb, bos_idx, eos_idx, max_len=30):
        device = img_emb.device
        B = img_emb.size(0)
        memory = self.img_proj(img_emb).unsqueeze(1)  # [B, 1, n_embd]

        # 以 <BOS> 开始生成
        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        outs = []

        for _ in range(max_len):
            attn_mask = (ys != self.pad_idx).long()  # 生成注意力掩码
            output = self.gpt2(input_ids=ys,
                               attention_mask=attn_mask,
                               encoder_hidden_states=memory)
            logits = output.logits[:, -1, :]  # 获取最新的logits
            nxt = logits.argmax(-1, keepdim=True)  # 预测下一个token
            outs.append(nxt)
            ys = torch.cat([ys, nxt], dim=1)  # 将新的token添加到输入序列中
            if (nxt.squeeze(1) == eos_idx).all():
                break

        return torch.cat(outs, dim=1)  # 返回生成的token序列