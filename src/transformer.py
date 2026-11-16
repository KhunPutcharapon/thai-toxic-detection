"""M3: เครื่องมือสำหรับ fine-tune WangchanBERTa (ถูกเรียกใช้จาก train_m3.py)

หน้าที่ของไฟล์นี้: มีแค่ "เครื่องมือ" — ไม่ได้ตัดสินใจว่าจะใช้ lr ไหน ฝึกกี่รอบ หรือกี่ seed
(การตัดสินใจและลำดับขั้นอยู่ใน train_m3.py) เทียบได้กับ src/linear.py ของ B0/B1

ข้อมูลไหลผ่านไฟล์นี้ตามลำดับ:
  ข้อความดิบ --preprocess_for_bert()--> ข้อความสะอาด --encode()--> เลข ID 128 ตัว
             --train_one()--> โมเดลที่ฝึกแล้ว --predict_proba()--> ความน่าจะเป็น toxic ต่อทวีต

ส่วนประกอบ:
  1. เตรียมข้อความ   preprocess_for_bert, preprocess_texts_bert
  2. ตั้งค่าพื้นฐาน    set_seed (ล็อกความสุ่ม), get_device (cuda / mps / cpu)
  3. โมเดล/tokenizer  load_tokenizer, load_model, encode
  4. ฝึก / ทาย        train_one, predict_proba
  5. reproducibility environment_info_bert (เวอร์ชัน torch/transformers/GPU)

ต่างจาก B0/B1: ไม่ตัดคำด้วย Deepcut และไม่ลบคำหยุด — ให้ tokenizer ของโมเดลหั่น subword เอง
(คำหยุดอย่าง "ไม่" เป็นบริบทสำคัญ) ส่วนการทำความสะอาดระดับข้อความใช้ฟังก์ชันเดียวกับ B0/B1

ใช้ PyTorch loop ธรรมดาแทน HF Trainer เพราะชื่อ argument ของ Trainer เปลี่ยนบ่อยระหว่างเวอร์ชัน
"""
import random
import re

import numpy as np

from . import config
from .preprocess import URL, USER, squeeze_repeats


# =====================================================================
# 1. เตรียมข้อความ
# =====================================================================

# ทำความสะอาดระดับข้อความแบบเดียวกับ B0/B1 (ลบลิงก์, @ผู้ใช้, ย่อคำยืด) แล้วแปลงให้ตรงกับตอน pretrain ของ WangchanBERTa
# (ตัวพิมพ์เล็ก เพราะเป็นโมเดล uncased และแทนช่องว่างด้วย <_> ตามที่ pythainlp ใช้กับโมเดลนี้)
def preprocess_for_bert(text):
    text = squeeze_repeats(USER.sub(" ", URL.sub(" ", str(text))))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.replace(" ", "<_>")


# ใช้ preprocess_for_bert กับทุกทวีต
def preprocess_texts_bert(texts):
    return [preprocess_for_bert(t) for t in texts]


# =====================================================================
# 2. ตั้งค่าพื้นฐาน
# =====================================================================

# ล็อกความสุ่มทุกไลบรารี (Python, numpy, torch) -> seed เดิมได้ผลเดิม
def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# เลือกอุปกรณ์อัตโนมัติ: cuda (Colab) > mps (Mac M-series) > cpu — ไฟล์เดียวรันได้ทุกที่
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():   # Mac M-series
        return torch.device("mps")
    return torch.device("cpu")


# =====================================================================
# 3. โมเดล / tokenizer
# =====================================================================

# tokenizer ของ WangchanBERTa (ตั้งค่าแบบเดียวกับที่ pythainlp ใช้) — หั่นข้อความเป็น subword
def load_tokenizer(model_name=config.M3_MODEL):
    from transformers import CamembertTokenizer
    tok = CamembertTokenizer.from_pretrained(model_name, revision="main")
    tok.additional_special_tokens = ["<s>NOTUSED", "</s>NOTUSED", "<_>"]
    return tok


# โหลด WangchanBERTa + ต่อหัวจำแนก 2 คลาส (หัวใหม่เริ่มจากค่าสุ่ม = "MISSING" ใน LOAD REPORT,
# หัว pretrain ที่ไม่ใช้ถูกทิ้ง = "UNEXPECTED" — ทั้งสองอย่างปกติ)
def load_model(model_name=config.M3_MODEL):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(model_name, revision="main", num_labels=2)


# แปลงข้อความเป็น tensor ครั้งเดียว (ข้อมูลเล็ก เก็บใน memory ได้ทั้งหมด)
# ทุกทวีตเติม/ตัดให้ยาว max_len token และนับว่ามีทวีตถูกตัดทิ้งกี่ทวีต (ควรเป็น 0)
def encode(tok, texts, labels=None, max_len=config.M3_MAX_LEN):
    import torch
    from torch.utils.data import TensorDataset
    enc = tok(texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    tensors = [enc["input_ids"], enc["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor(np.asarray(labels), dtype=torch.long))
    n_truncated = int(sum(len(tok(t)["input_ids"]) > max_len for t in texts))
    return TensorDataset(*tensors), n_truncated


# =====================================================================
# 4. ฝึก / ทาย
# =====================================================================

# ทายความน่าจะเป็นของคลาส toxic ทีละ batch (eval = ปิด dropout, no_grad = ไม่คำนวณ gradient)
def predict_proba(model, dataset, device, batch_size=64):
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            ids, mask = batch[0].to(device), batch[1].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# fine-tune 1 ครั้ง: คืน (model, ประวัติ) — ถ้าให้ dev มาจะวัด macro-F1 บน dev ทุก epoch
#   ใช้ 2 แบบ: มี dev = ขั้นเลือกพารามิเตอร์ / ไม่มี dev = ขั้น refit บนชุดฝึกทั้งหมด
#   ส่วนประกอบ: AdamW + weight decay (ลด overfit), warmup 10% แล้วลด lr เชิงเส้น
#   (กันการปรับแรงช่วงแรกจนลืมความรู้เดิม), clip gradient 1.0 (กันการฝึกระเบิด)
def train_one(train_ds, epochs, lr, seed, device, dev_ds=None, dev_y=None,
              batch_size=config.M3_BATCH, log=print):
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    model = load_model().to(device)                                       # เริ่มจากโมเดลตั้งต้นทุกครั้ง
    g = torch.Generator().manual_seed(seed)                               # ลำดับ batch คงที่ตาม seed
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    no_decay = ("bias", "LayerNorm.weight")                               # ไม่ใช้ weight decay กับ bias/LayerNorm (ธรรมเนียม BERT)
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": config.M3_WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(config.M3_WARMUP * total), total)

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        rec = dict(epoch=ep, train_loss=float(np.mean(losses)))
        if dev_ds is not None:
            dev_pred = (predict_proba(model, dev_ds, device) >= 0.5).astype(int)
            rec["dev_macro_f1"] = float(f1_score(dev_y, dev_pred, average="macro"))
        history.append(rec)
        log("    " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    return model, history


# =====================================================================
# 5. reproducibility — เวอร์ชันไลบรารีและ GPU (บันทึกลง metrics.json)
# =====================================================================
def environment_info_bert():
    import torch
    import transformers
    return dict(torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
