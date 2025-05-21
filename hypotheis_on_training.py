import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
from torchvision import transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from typing import List, Optional, Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class VisionEncoder(nn.Module):
    def __init__(self, d_model: int, freeze: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.backbone.hidden_dim, d_model, bias=False)
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        return self.proj(feats).unsqueeze(1)

class SwiGLU(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    def forward(self, x):
        return F.silu(x) * torch.sigmoid(self.beta * x)

class RotaryPosEmb(nn.Module):
    def __init__(self, dim: int, max_seq: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None], emb.sin()[None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)

class EnhancedMultiModalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 2048,
        max_len: int = 512,
        init_gain: float = 2.0,
        use_vision: bool = True
    ):
        super().__init__()
        self.use_vision = use_vision
        if use_vision:
            self.vision_encoder = VisionEncoder(d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = RotaryPosEmb(d_model, max_len)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff,
            activation=SwiGLU().forward,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.safety_head = nn.Linear(d_model, 2)
        self.max_len = max_len
        self.init_gain = init_gain
        self.use_checkpoint = True
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=self.init_gain)

    def forward(self, x: torch.LongTensor, images: Optional[torch.Tensor]=None):
        b, t = x.size()
        emb = self.tok_emb(x)
        emb = emb + self.pos_emb(emb)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        if self.use_vision and images is not None:
            img_emb = self.vision_encoder(images)
            memory = torch.cat([img_emb, emb], dim=1)
        else:
            memory = emb
        if self.use_checkpoint:
            h = checkpoint.checkpoint(self.decoder, emb, memory, mask)
        else:
            h = self.decoder(tgt=emb, memory=memory, tgt_mask=mask)
        h_norm = self.ln(h)
        logits = self.head(h_norm)
        safety_logits = self.safety_head(h_norm[:,0])
        return logits, safety_logits

    def gradient_thought_refinement(self, input_ids: torch.LongTensor, span: Tuple[int,int], steps: int=3, lr: float=1e-2):
        embed = self.tok_emb(input_ids)
        thought = embed[:, span[0]:span[1]].detach().clone().requires_grad_(True)
        rest = embed[:, span[1]:]
        for _ in range(steps):
            seq = torch.cat([thought, rest], dim=1)
            logits, _ = self.forward(seq)
            logp = F.log_softmax(logits[:, -1], dim=-1)
            loss = -logp.max(dim=-1)[0].mean()
            loss.backward()
            with torch.no_grad():
                thought -= lr * thought.grad
                thought.grad.zero_()
        return thought.detach()

class JSONDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int=512):
        with open(path, 'r') as f:
            data = json.load(f)
        self.examples = []
        for ex in data:
            inp = f"<s_thought> {ex['instruction']} <s_answer>"
            tgt = ex['response']
            seq = tokenizer.encode(inp + " " + tgt).ids
            if len(seq) > max_length:
                seq = seq[:max_length]
            safety = ex.get('safety_label', -1)
            img_path = ex.get('image_path', None)
            self.examples.append((torch.tensor(seq[:-1]), torch.tensor(seq[1:]), img_path, ex['instruction'], ex['response'], safety))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        inp, tgt, img_path, instr, resp, safety = self.examples[i]
        img = None
        if img_path:
            transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
            img = transform(Image.open(img_path).convert('RGB'))
        return inp, tgt, img, instr, resp, safety

def collate_batch(batch):
    inps, tgts, imgs, _, _, safeties = zip(*batch)
    inps = nn.utils.rnn.pad_sequence(inps, batch_first=True, padding_value=0)
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    if any(img is not None for img in imgs):
        imgs = torch.stack([i if i is not None else torch.zeros(3,224,224) for i in imgs])
    else:
        imgs = None
    safeties = torch.tensor(safeties)
    return inps, tgts, imgs, safeties

def train(
    model, dataloader, tokenizer,
    epochs: int=5, lr: float=1e-4, weight_decay: float=1e-6,
    device: str='cuda', warmup_steps: int=500, max_grad_norm: float=1.0,
    ct_dropout: float=0.1, high_weight: float=2.0, label_smooth_eps: float=0.1, safety_weight: float=1.0
):
    model = model.to(device)
    scaler = amp.GradScaler()
    opt = AdamW([
        {'params': [p for p in model.parameters() if p.dim()>1], 'weight_decay': weight_decay},
        {'params': [p for p in model.parameters() if p.dim()<=1], 'weight_decay': 0.0}
    ], lr=lr)
    total_steps = epochs * len(dataloader)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    crit = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    seen = set(ex['instruction'] for ex in json.load(open('train.json','r')))
    for e in range(epochs):
        for inp, tgt, imgs, safety_lbl in dataloader:
            inp, tgt = inp.to(device), tgt.to(device)
            imgs = imgs.to(device) if imgs is not None else None
            safety_lbl = safety_lbl.to(device)
            B, T = inp.size()
            is_known = torch.tensor([i in seen for i in ["" for _ in range(B)]], device=device)
            weight_ct = torch.where(is_known, torch.ones_like(is_known), torch.full_like(is_known, high_weight))
            with amp.autocast():
                logits, safety_logits = model(inp, imgs)
                loss_raw = crit(logits.view(-1, logits.size(-1)), tgt.view(-1))
                weights = torch.ones(B, T, device=device)
                for b in range(B):
                    idx = (inp[b]==tokenizer.vocab['<s_answer>']).nonzero(as_tuple=False)
                    end = idx[0,0] if idx.numel()>0 else T
                    weights[b,:end] = weight_ct[b]
                if torch.rand(1).item() < ct_dropout:
                    weights.fill_(1.0)
                tgt_onehot = F.one_hot(tgt, logits.size(-1)).float()
                tgt_smooth = tgt_onehot*(1-label_smooth_eps) + label_smooth_eps/logits.size(-1)
                logp = F.log_softmax(logits, dim=-1)
                loss_task = (-(tgt_smooth*logp).sum(-1)*weights).sum()/weights.sum()
                if safety_lbl.min()>=0:
                    loss = loss_task + safety_weight*F.cross_entropy(safety_logits, safety_lbl)
                else:
                    loss = loss_task
            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
        print(f"Epoch {e+1}/{epochs} loss {loss.item():.4f}")

if __name__ == '__main__':
    raw = json.load(open('train.json','r'))
    # train BPE tokenizer
    def get_corpus():
        for ex in raw:
            yield f"<s_thought> {ex['instruction']} <s_answer> {ex['response']}"
    bpe = Tokenizer(BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=30000,
        min_frequency=2,
        special_tokens=["<pad>","<unk>","<s_thought>","<s_answer>"]
    )
    bpe.train_from_iterator(get_corpus(), trainer=trainer)
    bpe.save("tokenizer.json")
    tok = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<s_thought>","<s_answer>"]
    )
    tok.vocab = tok.get_vocab()
    ds = JSONDataset('train.json', tok)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_batch, num_workers=4, pin_memory=True)
    model = EnhancedMultiModalTransformerLM(len(tok.vocab))
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","out_proj"]
    )
    model = get_peft_model(model, peft_cfg)
    train(model, dl, tok)
