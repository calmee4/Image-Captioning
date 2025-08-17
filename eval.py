# eval.py
import os
from collections import defaultdict
from PIL import Image
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from dataset import Vocab, simple_tokenize  # 用到你的 tokenizer

def _ids_to_tokens(ids_tensor, vocab):
    toks = []
    for tid in ids_tensor.tolist():
        w = vocab.itos[int(tid)]
        if w == Vocab.EOS: break
        if w in (Vocab.PAD, Vocab.BOS): continue
        toks.append(w)
    return toks

@torch.no_grad()
def eval_bleu(encoder, decoder, data_loader, vocab, device, max_len=30):
    """
    正确的“按图聚合 5 参考”的 BLEU 评测。
    只要 data_loader 是你的 test_loader 即可；我们直接读取其 dataset 来聚合。
    """
    encoder.eval(); decoder.eval()
    ds = data_loader.dataset           # CaptionDataset
    pairs = ds.pairs                   # [{"image": fn, "caption": str}, ...]
    images_root = ds.images_root
    transform = ds.transform           # eval transform

    # 1) 聚合每张图的参考（tokens）
    refs_by_img = defaultdict(list)
    for it in pairs:
        refs_by_img[it["image"]].append(simple_tokenize(it["caption"]))

    # 2) 对每张图只生成一次预测
    hyps = []   # list[str]
    refs = []   # list[list[str]] (多参考)
    smoothie = SmoothingFunction().method1

    for fn, ref_tok_lists in refs_by_img.items():
        img = Image.open(os.path.join(images_root, fn)).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        img_emb = encoder(img)
        hyp_ids = decoder.greedy_decode(img_emb, vocab.bos, vocab.eos, max_len=max_len)[0]
        hyp_tok = [w for w in vocab.decode(hyp_ids.tolist())
                   if w not in (Vocab.PAD, Vocab.BOS, Vocab.EOS)]
        hyps.append(" ".join(hyp_tok))
        refs.append([" ".join(toks) for toks in ref_tok_lists])

    # 3) NLTK 的 corpus_bleu 需要 refs 维度为 [num_refs][num_imgs]
    refs_T = list(map(list, zip(*refs)))

    # hyps: [num_imgs][tokens]，已经是 list[str] 或 list[list[str]]
    # refs: [num_imgs][num_refs][tokens]

    smoothie = SmoothingFunction().method1

    b1 = corpus_bleu(refs, hyps,
                        weights=(1, 0, 0, 0),
                        smoothing_function=smoothie) * 100
    b2 = corpus_bleu(refs, hyps,
                        weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smoothie) * 100
    b3 = corpus_bleu(refs, hyps,
                        weights=(1 / 3, 1 / 3, 1 / 3, 0),
                        smoothing_function=smoothie) * 100
    b4 = corpus_bleu(refs, hyps,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothie) * 100


    print(f"Test BLEU-1/2/3/4 = {b1:.2f} / {b2:.2f} / {b3:.2f} / {b4:.2f}")
    return {"BLEU-1":b1, "BLEU-2":b2, "BLEU-3":b3, "BLEU-4":b4}

@torch.no_grad()
def show_samples(encoder, decoder, data_loader, vocab, device, n=5, max_len=30):
    encoder.eval(); decoder.eval()
    ds = data_loader.dataset
    pairs = ds.pairs
    images_root = ds.images_root
    transform = ds.transform

    # 聚合参考
    from collections import defaultdict
    refs_by_img = defaultdict(list)
    for it in pairs:
        refs_by_img[it["image"]].append(it["caption"])
    items = list(refs_by_img.items())[:n]

    for fn, ref_list in items:
        img = Image.open(os.path.join(images_root, fn)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        hyp_ids = decoder.greedy_decode(encoder(img_t), vocab.bos, vocab.eos, max_len=max_len)[0].tolist()
        hyp = " ".join([w for w in vocab.decode(hyp_ids) if w not in (Vocab.PAD, Vocab.BOS, Vocab.EOS)])

        print(f"\nImage: {fn}")
        print("PR:", hyp)
        for i, r in enumerate(ref_list):
            print(f"GT#{i}:", r)
