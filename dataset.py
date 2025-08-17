import os
import json
import re
from collections import Counter
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as T
import torch
from torch.nn.utils.rnn import pad_sequence

def get_split(data,name):
    return [img for img in data["images"] if img.get("split")== name ]

def flatten_pairs(images): #得到 {image , caption}对
    for img in images:
        fn = img["filename"]
        for s in img.get("sentences", []): # 注意拼写：sentences
            cap = s.get("raw") or " ".join(s.get("tokens", []))
            yield {"image": fn, "caption": cap}


def simple_tokenize(s: str):
    s = s.lower().strip()  # 转为小写并去掉两端空格
    return [w for w in re.split(r"[^\w]+", s) if w]  # 正则分词，去除非字母数字字符


# 词汇表类
class Vocab:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, min_freq=2, max_size=20000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK: 3}  # stoi = string to index
        self.itos = [self.PAD, self.BOS, self.EOS, self.UNK]  # itos = index to string

    def build_from_pairs(self, pairs):
        """
        从图像描述对（pairs）中构建词汇表
        """
        freq = Counter()  # 统计词频
        for pair in pairs:
            for word in simple_tokenize(pair["caption"]):
                freq[word] += 1

        # 按照词频排序，并过滤掉频率过低的词
        words = [w for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])) if c >= self.min_freq]

        # 限制词汇表大小
        if self.max_size:
            words = words[: self.max_size - len(self.itos)]

        # 添加到词汇表
        for word in words:
            self.stoi[word] = len(self.itos)
            self.itos.append(word)

    def encode(self, tokens):
        """
        将单词列表转为数字 ID 列表
        """
        return [self.stoi.get(w, self.stoi[self.UNK]) for w in tokens]

    def decode(self, ids):
        """
        将数字 ID 列表转回单词列表
        """
        return [self.itos[i] for i in ids]

    @property
    def pad(self):
        return self.stoi[self.PAD]

    @property
    def bos(self):
        return self.stoi[self.BOS]

    @property
    def eos(self):
        return self.stoi[self.EOS]

    def __len__(self):
        return len(self.itos)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# 数据增强使用的transform
def build_train_transform(img_size=224):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        # 颜色扰动轻一点，别破坏语义
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
def build_eval_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

#创建torch.Dataset
class CaptionDataset(Dataset):
    def __init__(self, pairs, images_root, vocab,transform=None,max_len=40):
        self.pairs = pairs
        self.images_root = images_root
        self.vocab = vocab
        self.transform = transform or build_eval_transform()
        self.max_len = max_len
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx): #取出图片和idx 句子编号
        item = self.pairs[idx]
        img_path = os.path.join(self.images_root,item["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        tokens = simple_tokenize(item["caption"])
        if self.max_len is not None and len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len-2]
        ids = [self.vocab.bos] + self.vocab.encode(tokens) + [self.vocab.eos]
        ids = torch.tensor(ids,dtype=torch.long)
        # 返回的是图片和token词元
        return image, ids

#图像增强
def collate_fn(batch):
    """
    batch: List[(image[C,H,W], ids[T])]
    returns:
      images: [B, C, H, W]
      padded: [B, T_max]
      lengths: [B]
    """
    # pad所有的token
    images, seqs = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)  # 0 对应 <pad>
    return images, padded, lengths

#接口函数加载数据
def load_data(base_dir, batch_size=32):

    json_path = os.path.join(base_dir, "data/dataset_flickr8k.json")
    images_root = os.path.join(base_dir, "data/images")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = get_split(data, "train")
    val_data   = get_split(data, "val")
    test_data  = get_split(data, "test")

    train_pairs = list(flatten_pairs(train_data))
    val_pairs   = list(flatten_pairs(val_data))
    test_pairs  = list(flatten_pairs(test_data))

    vocab = Vocab(min_freq=2, max_size=20000)
    vocab.build_from_pairs(train_pairs)

    train_tf = build_train_transform(224)
    eval_tf  = build_eval_transform(224)

    train_ds = CaptionDataset(train_pairs, images_root, vocab, transform=train_tf,  max_len=40)
    val_ds   = CaptionDataset(val_pairs,   images_root, vocab, transform=eval_tf,   max_len=40)
    test_ds  = CaptionDataset(test_pairs,  images_root, vocab, transform=eval_tf,   max_len=40)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, vocab
