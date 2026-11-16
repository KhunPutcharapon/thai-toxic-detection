"""M3: fine-tune WangchanBERTa — ชุดฝึก/ชุดทดสอบ/เมตริกเดียวกับ B0, B1

=====================================================================
ภาพรวม
=====================================================================
ไฟล์นี้เป็น "ตัวคุมขั้นตอน" ของ M3 ส่วนเครื่องมือจริง (เตรียมข้อความ, โหลดโมเดล, ฝึก, ทาย)
อยู่ใน src/transformer.py  — เหมือนที่ train_b1.py เรียกใช้ src/linear.py
  ขั้นที่ 1  เลือก lr กับ epoch: แบ่ง 10% ของชุดฝึกเป็น dev -> ลอง lr 3 ค่า x 5 epoch
             -> เลือกคู่ที่ dev macro-F1 สูงสุด (ไม่แตะชุดทดสอบ)
  ขั้นที่ 2  refit: ฝึกใหม่บนชุดฝึกทั้งหมด 1,472 ทวีตด้วยค่าที่เลือก ซ้ำ 5 seeds
             -> ทายชุดทดสอบ 176 ทวีต seed ละครั้ง
  ผลลัพธ์    results/m3/              = ผลของ seed 42 (ใช้กับ compare.py)
                                        + metrics.json มี seed_summary (mean ± SD ทุก seed)
             results/m3/seeds/seed_XX/ = ผลแยกของแต่ละ seed
             ไฟล์ในแต่ละโฟลเดอร์: metrics.json, predictions.csv, errors.csv, probs.csv

=====================================================================
วิธีรันบน Mac (Apple Silicon ใช้ GPU ผ่าน MPS)
=====================================================================
ต้องใช้ Python >= 3.10 (torch/transformers รุ่นใหม่ไม่รองรับ 3.9 ที่ใช้กับ Deepcut)
และใช้ venv แยกจาก venv_deepcut

  [ครั้งแรกครั้งเดียว] สร้าง venv + ลงไลบรารี
    cd ~/Desktop/"CS374(NLP)_Project"/thai-toxic-detection
    brew install python@3.12                       # ถ้ายังไม่มี (python3 ของระบบเป็น 3.9)
    python3.12 -m venv venv_m3
    source venv_m3/bin/activate                    # ต้นบรรทัดจะขึ้น (venv_m3)
    pip install --upgrade pip
    pip install numpy pandas scikit-learn scipy -r requirements_m3.txt
    python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
                                                   # ต้องได้ MPS: True

  [ทุกครั้ง] เปิด venv
    cd ~/Desktop/"CS374(NLP)_Project"/thai-toxic-detection
    source venv_m3/bin/activate

  1) smoke test — เช็กว่าโค้ดรันผ่าน (~1 นาที; ครั้งแรกโหลดโมเดล ~420 MB)
    python train_m3.py --smoke                     # ผลไป results/m3_smoke/ (ตัวเลขไม่มีความหมาย)

  2) รันจริง (~20–45 นาทีบน M5 — ประมาณจาก smoke test)
    caffeinate -i python train_m3.py 2>&1 | tee results_m3_log.txt
      caffeinate -i  = กัน Mac หลับระหว่างรัน (เสียบสายชาร์จไว้)
      tee            = แสดงบนจอ + บันทึกเป็น log ไว้ใส่ final package

  3) เทียบผลกับ baseline + เก็บ log
    python compare.py --a results/b1 --b results/m3
    python compare.py --a results/b0 --b results/m3
    mv results_m3_log.txt results/m3/train_log.txt

  4) ปิด venv:  deactivate

  * ต้องรัน B0/B1 ให้เสร็จก่อน (bash run_baselines.sh ใน venv_deepcut) เพื่อให้มีผลไว้เทียบ

=====================================================================
ตัวเลือกเพิ่มเติม
=====================================================================
  python train_m3.py --lr 3e-5 --epochs 3          ข้ามขั้นที่ 1 ใช้ค่าที่กำหนดเอง
  python train_m3.py --seeds 42 43 44              กำหนด seed เอง (ค่าเริ่มต้น 42–46)
  ค่าอื่น (batch, max_len, lr grid, dev 10%) แก้ที่ src/config.py ส่วน "M3"

=====================================================================
ทางเลือก: Google Colab
=====================================================================
  เปิด m3_colab.ipynb -> Runtime > Change runtime type > T4 GPU -> รันทีละเซลล์
  (ต้อง push โค้ดขึ้น GitHub ก่อน หรืออัปโหลด zip ของโฟลเดอร์โดยไม่รวม venv_*)

=====================================================================
ปัญหาที่อาจเจอ
=====================================================================
  ModuleNotFoundError                 ลืม source venv_m3/bin/activate
  ... requires the protobuf library   pip install protobuf  (อย่าลง tiktoken ตามที่ error แนะนำ)
  ... not implemented for MPS         PYTORCH_ENABLE_MPS_FALLBACK=1 python train_m3.py
  memory เต็ม / เครื่องค้าง             ลด M3_BATCH = 8 ใน src/config.py
  LOAD REPORT: UNEXPECTED / MISSING   ปกติ — ทิ้งหัว pretrain (lm_head) และสร้างหัวจำแนกใหม่ (classifier)
  คำเตือน HF_TOKEN / Redirects        ไม่ต้องสนใจ
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data import load_splits
from src.evaluate import METRIC_NAMES, compute_metrics, environment_info, print_metrics, save_run
from src.transformer import (encode, environment_info_bert, get_device, load_tokenizer,
                             predict_proba, preprocess_texts_bert, train_one)

# ---------------------------------------------------------------------
# ค่าคงที่
# ---------------------------------------------------------------------
OUT_DIR = config.RESULTS_DIR / "m3"
SUMMARY_KEYS = [k for k, _ in METRIC_NAMES] + ["fp", "fn"]     # เมตริกที่คิด mean ± SD ข้าม seed


# ---------------------------------------------------------------------
# ฟังก์ชันช่วย
# ---------------------------------------------------------------------


# ลบโมเดลเก่าออกจาก memory ของ GPU (ทั้งสคริปต์ฝึก 8 ครั้ง ถ้าไม่ลบ memory จะเต็ม)
def free(model):
    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ขั้นที่ 1: เลือก lr และจำนวน epoch จาก dev (ใช้ seed หลักตัวเดียว)
# ได้ 15 ตัวเลือก (3 lr x 5 epoch) -> เลือก dev macro-F1 สูงสุด; เท่ากันเลือก epoch น้อย แล้ว lr น้อย
def select_params(tok, train, device, lr_grid, max_epochs):
    sub, dev = train_test_split(train, test_size=config.M3_DEV_FRAC, stratify=train.is_toxic,
                                random_state=config.SEED)
    print(f"[เลือกพารามิเตอร์] ฝึก {len(sub)} / dev {len(dev)} ทวีต (ไม่แตะชุดทดสอบ)")
    sub_ds, _ = encode(tok, sub.text.tolist(), sub.is_toxic.values)
    dev_ds, _ = encode(tok, dev.text.tolist())
    log = []
    for lr in lr_grid:
        print(f"  lr={lr:g}")
        model, hist = train_one(sub_ds, max_epochs, lr, config.SEED, device,
                                dev_ds=dev_ds, dev_y=dev.is_toxic.values)
        free(model)
        log += [dict(lr=lr, **h) for h in hist]
    # ดีที่สุดตาม dev macro-F1; ถ้าเท่ากันเลือก epoch น้อยกว่า แล้ว lr น้อยกว่า
    best = sorted(log, key=lambda r: (-r["dev_macro_f1"], r["epoch"], r["lr"]))[0]
    print(f"[เลือกได้] lr={best['lr']:g}  epochs={best['epoch']}  (dev macro-F1 {best['dev_macro_f1']:.4f})")
    return best["lr"], best["epoch"], log


# ขั้นที่ 2: ฝึกบนชุดฝึกทั้งหมด (รวม dev กลับเข้าไป ให้เห็นข้อมูลเท่า B0/B1) ด้วย seed หนึ่ง แล้วทายชุดทดสอบ
# ตัดสิน toxic ที่ความน่าจะเป็น >= 0.5 แล้ววัดผลด้วย compute_metrics ตัวเดียวกับ B0/B1
def refit_and_test(tok, train, test, lr, epochs, seed, device):
    train_ds, _ = encode(tok, train.text.tolist(), train.is_toxic.values)
    test_ds, _ = encode(tok, test.text.tolist())
    t0 = time.time()
    model, hist = train_one(train_ds, epochs, lr, seed, device)
    prob = predict_proba(model, test_ds, device)
    free(model)
    pred = (prob >= 0.5).astype(int)
    m = compute_metrics(test.is_toxic.values, pred)
    m.update(seed=seed, train_history=hist, train_seconds=round(time.time() - t0, 1))
    return m, pred, prob


# บันทึกความน่าจะเป็นรายทวีต (B0/B1 ไม่มี) ไว้ทำ error analysis: ผิดแบบมั่นใจ หรือแค่ลังเล
def save_probs(out_dir, test, prob):
    test[["tweet_id", "is_toxic", "keyword"]].assign(prob_toxic=prob) \
        .to_csv(out_dir / "probs.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# ลำดับการทำงานทั้งหมด
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="ทดสอบโค้ดเร็ว ๆ (ผลไม่มีความหมาย)")
    ap.add_argument("--lr", type=float, help="กำหนด lr เอง (ข้ามขั้นเลือกพารามิเตอร์ ต้องใส่ --epochs ด้วย)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=config.M3_SEEDS)
    args = ap.parse_args()

    # --- อ่านข้อมูล (ชุดเดียวกับ B0/B1) + โหมด smoke ใช้ข้อมูลนิดเดียว ผลแยกโฟลเดอร์ ---
    train, test = load_splits()
    out_dir, lr_grid, max_epochs, seeds = OUT_DIR, config.M3_LR_GRID, config.M3_MAX_EPOCHS, args.seeds
    if args.smoke:
        train = pd.concat([train[train.is_toxic == lab].head(40) for lab in (1, 0)], ignore_index=True)
        out_dir, lr_grid, max_epochs, seeds = config.RESULTS_DIR / "m3_smoke", lr_grid[:1], 1, seeds[:1]
    print(f"[ข้อมูล] ชุดฝึก {len(train)} ทวีต / ชุดทดสอบ {len(test)} ทวีต")

    # --- เลือกอุปกรณ์: cuda (Colab) / mps (Mac) / cpu ---
    device = get_device()
    print(f"[อุปกรณ์] {device}" + ("  ⚠ ไม่มี GPU จะช้ามาก" if device.type == "cpu" and not args.smoke else ""))

    # --- เตรียมข้อความ + sanity check ว่า tokenizer หั่นถูก และไม่มีทวีตถูกตัดทิ้ง ---
    train["text"] = preprocess_texts_bert(train["tweet_text"])
    test["text"] = preprocess_texts_bert(test["tweet_text"])
    tok = load_tokenizer()
    _, n_trunc_train = encode(tok, train.text.tolist())
    _, n_trunc_test = encode(tok, test.text.tolist())
    print(f"[tokenizer] ตัวอย่าง: {tok.tokenize(test.text.iloc[0])[:15]}")
    print(f"[tokenizer] ทวีตที่ยาวเกิน {config.M3_MAX_LEN} token: ชุดฝึก {n_trunc_train} / ชุดทดสอบ {n_trunc_test}")

    # --- ขั้นที่ 1: เลือก lr / epoch (หรือใช้ค่าที่กำหนดเองผ่าน --lr --epochs) ---
    if args.lr is not None:
        assert args.epochs, "ใส่ --epochs คู่กับ --lr"
        lr, epochs, selection = args.lr, args.epochs, "manual"
    else:
        lr, epochs, selection = select_params(tok, train, device, lr_grid, max_epochs)

    # --- ขั้นที่ 2: refit ทีละ seed (ทุก seed ใช้ lr/epoch เดียวกัน ต่างกันแค่ความสุ่ม) ---
    runs = []
    for s in seeds:
        print(f"[refit] seed={s}  lr={lr:g}  epochs={epochs}  บนชุดฝึกทั้งหมด {len(train)} ทวีต")
        m, pred, prob = refit_and_test(tok, train, test, lr, epochs, s, device)
        seed_dir = out_dir / "seeds" / f"seed_{s}"
        save_run(seed_dir, test, pred, m)
        save_probs(seed_dir, test, prob)
        print(f"  macro-F1 {m['macro_f1']:.3f}  acc {m['acc']:.3f}  FP {m['fp']}  FN {m['fn']}  ({m['train_seconds']}s)")
        runs.append((m, pred, prob))

    # --- สรุป: ผลหลัก = seed แรก (42 ตรงกับ seed ของทั้งโปรเจกต์ เลือกไว้ล่วงหน้า ไม่ใช่เลือกตัวที่ผลดีสุด)
    #     + mean ± SD ของทุก seed (ddof=1 = SD แบบตัวอย่าง) เก็บใน metrics.json ---
    main_m, main_pred, main_prob = runs[0]
    summary = {k: dict(mean=float(np.mean([r[0][k] for r in runs])),
                       sd=float(np.std([r[0][k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0)
               for k in SUMMARY_KEYS}
    metrics = {k: v for k, v in main_m.items() if k != "train_history"}
    metrics.update(
        model="M3: WangchanBERTa fine-tune", hf_model=config.M3_MODEL,
        best_params=dict(lr=lr, epochs=epochs, batch_size=config.M3_BATCH, max_len=config.M3_MAX_LEN,
                         weight_decay=config.M3_WEIGHT_DECAY, warmup=config.M3_WARMUP),
        selection_log=selection, dev_frac=config.M3_DEV_FRAC, scoring="dev macro-F1",
        seeds=seeds, seed_summary=summary,
        n_truncated=dict(train=n_trunc_train, test=n_trunc_test),
        n_train=len(train), n_test=len(test), device=str(device),
        **environment_info(), **environment_info_bert())
    save_run(out_dir, test, main_pred, metrics)
    save_probs(out_dir, test, main_prob)

    # --- พิมพ์ผลออกจอ ---
    print(f"\n=== M3 บนชุดทดสอบ 176 ทวีต (seed {seeds[0]}) ===")
    print_metrics(metrics)
    print(f"\n=== mean ± SD จาก {len(seeds)} seeds ===")
    for key, name in METRIC_NAMES:
        print(f"  {name:18s}{summary[key]['mean']:.3f} ± {summary[key]['sd']:.3f}")
    print(f"  {'FP':18s}{summary['fp']['mean']:.1f} ± {summary['fp']['sd']:.1f}")
    print(f"\nบันทึกผลใน {out_dir}/  -> ต่อด้วย: python compare.py --a results/b1 --b {out_dir.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
