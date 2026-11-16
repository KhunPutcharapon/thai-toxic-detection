<a id="top"></a>

# เมื่อคำหยาบไม่ได้แปลว่า Toxic: เปรียบเทียบ Traditional NLP กับ Transformer ในการตรวจจับข้อความ Toxic ภาษาไทย

*Comparing Traditional NLP and Transformer Models for Thai Toxic Language Detection*

CS374/CS379 Natural Language Processing · ภาคการศึกษาที่ 1/2569

**สมาชิก:** นายชยากร จิตรอุดมกุล 6709616400 · นายชัชวีร์ ไทยเจริญ 6709616426 · นายพัชรพล คุณวโรตม์ 6709616715

## สารบัญ

- [📄 Proposal](#proposal)
- [💻 Code](#code)

---

<a id="proposal"></a>

## 📄 Proposal

<a id="p1"></a>

### 1. Motivation

ในทวีตภาษาไทย คำหยาบไม่ได้แปลว่าข้อความนั้น toxic เสมอไป คำเดียวกันอาจใช้ด่าทอคนอื่น หรือใช้เป็นคำเน้นอารมณ์ อุทาน หรือคุยเล่นกับเพื่อน ระบบที่ตัดสินจากการมีคำหยาบอย่างเดียวจึงเสี่ยงคัดกรองข้อความปกติทิ้งเกินจำเป็น

งานต้นฉบับของ Thai Toxicity Tweet Corpus ใช้ CountVectorizer (Bag-of-Words) + Logistic Regression และผู้เขียนระบุว่ายังไม่เพียงพอ ปัจจุบันมีโมเดลภาษาไทยแบบ Transformer อย่าง WangchanBERTa ที่ใช้บริบทของประโยคได้ แต่ยังไม่มีการเทียบสองแนวทางนี้บนการแบ่งข้อมูลเดียวกัน

**วัตถุประสงค์:** เปรียบเทียบ Traditional NLP (นับคำ / TF-IDF + Logistic Regression) กับ Transformer (WangchanBERTa) ในการจำแนกทวีตภาษาไทยเป็น Toxic / Non-Toxic โดยเน้นว่าแต่ละโมเดลแยกทวีตที่มีคำหยาบแต่ไม่ toxic ได้ดีแค่ไหน

<a id="p2"></a>

### 2. Problem Statement

โมเดลที่ใช้ feature แบบนับคำหรือ TF-IDF มีแนวโน้มเรียนรู้จาก **การมีหรือไม่มี keyword** เป็นหลัก จึงเสี่ยงทำนาย **ทวีตที่มีคำหยาบแต่ไม่ toxic ผิดเป็น toxic (false positive)** ยังไม่มีหลักฐานเชิงทดลองว่า เมื่อทวีตมี keyword เดียวกัน โมเดลแบบ traditional กับ Transformer ภาษาไทยแยกกรณีที่ toxic ออกจากกรณีที่ไม่ toxic ได้ต่างกันแค่ไหน ภายใต้ data split และเกณฑ์ประเมินเดียวกัน

**ขอบเขต:** ศึกษาเฉพาะทวีตที่มีคำ keyword ตามวิธีเก็บข้อมูลของชุดข้อมูล ไม่ได้มุ่งแก้ toxic แบบแฝงที่ไม่มีคำหยาบ เช่น การประชดหรือเหน็บแนม

<a id="p3"></a>

### 3. Related Work

| | รายละเอียด |
|---|---|
| **Paper** | Sirihattasak, Komachi & Ishikawa (2018). *Annotation and Classification of Toxicity for Thai Twitter.* TA-COS 2018 |
| **Data** | 3,300 ทวีต จากคำสำคัญ 44 คำ (คำละ 75), annotator 3 คน ใช้เสียงข้างมาก (toxic 2,027 / non-toxic 1,273) |
| **Method** | CountVectorizer (Bag-of-Words) + Logistic Regression, ตัดคำด้วย Deepcut |
| **Evaluation** | ชุดทดสอบ 176 ทวีต (44 คำ × 2 toxic + 2 non-toxic) จากทวีตที่ annotator เห็นตรงกัน; ชุดฝึก balanced 944/944 |
| **ผลลัพธ์** | Accuracy 0.80 · Precision 0.87 · Recall 0.70 · F1 0.78 · keyword baseline F1 0.67 |
| **ข้อจำกัดที่ผู้เขียนระบุ** | คำสแลง การตัดคำ และการตีความตามบริบท |

<a id="p4"></a>

### 4. Research Gap

1. งานเดิมใช้แค่การนับคำ + Logistic Regression และผู้เขียนระบุว่ายังไม่เพียงพอ
2. ยังไม่มีการเทียบ WangchanBERTa บนการแบ่งข้อมูลเดียวกับงานเดิม
3. งานเดิมรายงานคะแนนรวม แต่ไม่ได้วิเคราะห์ว่าโมเดลจัดการ **ทวีตที่มีคำหยาบแต่ไม่ toxic** ได้ดีแค่ไหน
4. งานเดิมใช้ชุดทดสอบที่ annotator เห็นตรงกันทั้งหมด จึงยังไม่รู้ว่าโมเดลทำงานอย่างไรกับข้อความกำกวม

**Contribution:** controlled comparison B0 → B1 → M3 บน protocol ที่เลียนแบบงานต้นฉบับ + วิเคราะห์ false positive บนทวีตที่มีคำหยาบแต่ไม่ toxic + pipeline ที่รันซ้ำได้

<a id="p5"></a>

### 5. Research Question และ Hypothesis

**RQ (หลัก):** การเปลี่ยนวิธีแทนข้อความจากการนับคำ (B0) เป็น TF-IDF (B1) และ WangchanBERTa (M3) ภายใต้ data split, seed และเกณฑ์ประเมินเดียวกัน ส่งผลต่อประสิทธิภาพและข้อผิดพลาดในการตรวจจับข้อความ toxic ภาษาไทยอย่างไร โดยเฉพาะกับทวีตที่มีคำหยาบแต่ไม่ toxic

- **H1:** M3 จะได้ macro-F1 สูงกว่า B0/B1 เพราะชุดทดสอบมีทั้ง toxic และ non-toxic ใน keyword เดียวกัน ซึ่งต้องอาศัยบริบทในการแยก
- **H2:** B0/B1 จะมี false positive rate บนทวีต non-toxic สูงกว่า M3 เพราะน้ำหนักของ keyword กลบบริบทรอบข้าง

<a id="p6"></a>

### 6. Dataset Protocol

| รายการ | งานเดิม | ของเรา |
|---|---|---|
| ข้อมูลต้นทาง | 3,300 ทวีต, 44 คำสำคัญ × 75 | ชุดเดียวกัน เหลือ 2,160 ทวีต (ข้อความหาย/ถูกลบ 1,140) |
| Label | เสียงข้างมาก 3 annotator (toxic 2,027 / non-toxic 1,273) | เหมือนกัน (toxic 1,332 / non-toxic 828) |
| ชุดทดสอบ | 176 ทวีต, คำละ 2+2, จาก gold | 176 ทวีต, คำละ 2+2, จากทวีตโหวต 3–0 |
| ชุดฝึก | 1,888 ทวีต สมดุล 944/944 | 1,472 ทวีต สมดุล 736/736 (คำละไม่เกิน 30 ต่อป้าย) |
| ตัวตัดคำ | Deepcut | Deepcut (B0/B1); M3 ใช้ tokenizer ของ WangchanBERTa |

- seed = 42 ทุกขั้น; บันทึกชุดฝึก/ชุดทดสอบเป็นไฟล์ใน `data/splits/` ให้ทุกโมเดลใช้ร่วมกัน
- ตรวจแล้วว่าไม่มีทวีตซ้ำ (ทั้ง id และข้อความ) ระหว่างชุดฝึกกับชุดทดสอบ
- แทน @mention และ URL ก่อนประมวลผล และก่อนยกตัวอย่างในรายงาน

<a id="p7"></a>

### 7. Baseline และ Main Comparison

| Condition | วิธี | บทบาท |
|---|---|---|
| Ref | ผลที่งานเดิมรายงาน | อ้างอิงเท่านั้น (ชุดข้อมูลไม่เท่ากัน) |
| **B0** | Bag-of-Words (CountVectorizer) + Logistic Regression | ทำซ้ำวิธีของงานเดิม |
| **B1** | TF-IDF + Logistic Regression | เปลี่ยนจาก B0 แค่การให้น้ำหนักคำ |
| **M3** | WangchanBERTa fine-tune สำหรับ binary classification | Transformer |

**ควบคุมให้เหมือนกัน:** ชุดฝึก/ชุดทดสอบ, seed, preprocessing ระดับข้อความ, evaluation script
B0/B1 เลือกพารามิเตอร์ด้วย 3-fold CV ภายในชุดฝึก (macro-F1); M3 tune learning rate / epochs / batch size และรันหลาย seed รายงาน mean ± SD

<a id="p8"></a>

### 8. Metrics และ Error Analysis Plan

- **Metrics:** Accuracy, Precision / Recall / F1 ของคลาส toxic, macro-F1, Confusion Matrix
- **ตอบ RQ โดยตรง:** false positive rate บนทวีต non-toxic 88 ทวีต (ทุกทวีตมี keyword) และ per-keyword accuracy
- **นัยสำคัญ:** McNemar exact test + bootstrap 95% CI ทีละคู่โมเดล (ชุดทดสอบ 176 ทวีตเล็ก)

| Error type | ลักษณะ |
|---|---|
| Profanity-as-emphasis FP | ใช้คำหยาบเน้นอารมณ์/อุทาน ไม่ได้ด่าใคร |
| Friendly banter FP | ใช้คำหยาบคุยเล่นกับเพื่อน |
| Missed toxic (FN) | มีคำหยาบและด่าจริง แต่ทำนายเป็น non-toxic |
| Segmentation / spelling | คำหยาบที่สะกดเลี่ยง/คำยืด ตัดคำผิด |

<a id="p9"></a>

### 9. Risks, Ethics, Limitations

- **Risks:** ข้อมูลหาย 1,140 ทวีต ชุดฝึกเล็กกว่างานเดิม → ผล M3 อาจแปรปรวน (แก้ด้วยหลาย seed); GPU จำกัด → ใช้ Colab + WangchanBERTa base
- **Ethics:** ข้อมูลจากผู้ใช้จริง ไม่เผยแพร่ชื่อผู้ใช้; label มาจากผู้ประเมิน 3 คน อาจมี bias; ระบบอาจถูกใช้เซ็นเซอร์เกินจำเป็น
- **Limitations:** เฉพาะทวีตที่มี keyword ไม่ครอบคลุมการประชด/เหน็บแนม; domain เดียว; ชุดข้อมูลไม่เท่ากับงานเดิม จึงเทียบตัวเลขงานเดิมได้เชิงอ้างอิง; ชุดทดสอบ 176 ทวีตเล็ก

<a id="p10"></a>

### 10. Reproducibility และความคืบหน้า

โค้ด, ไฟล์ split, seed, config และเวอร์ชันไลบรารีอยู่ในส่วน [💻 Code](#code) ทุกโมเดลบันทึกผลรูปแบบเดียวกันใน `results/`

**ผลปัจจุบัน (ชุดทดสอบ 176 ทวีต)**

| Metric | งานเดิม | B0 (นับคำ) | B1 (TF-IDF) | M3 seed 42 | M3 mean ± SD (5 seeds) |
|---|---|---|---|---|---|
| Precision (toxic) | 0.87 | 0.679 | 0.690 | 0.840 | 0.828 ± 0.017 |
| Recall (toxic) | 0.70 | 0.602 | 0.682 | 0.898 | 0.875 ± 0.027 |
| F1 (toxic) | 0.78 | 0.639 | 0.686 | 0.868 | 0.851 ± 0.015 |
| Accuracy | 0.80 | 0.659 | 0.688 | 0.864 | 0.847 ± 0.014 |
| macro-F1 | – | 0.658 | 0.687 | 0.863 | 0.846 ± 0.015 |
| FP (non-toxic ที่ถูกทายเป็น toxic, จาก 88) | – | 25 | 27 | 15 | 16.0 ± 2.0 |

- B1 − B0: accuracy +0.028, McNemar p = 0.405, bootstrap 95% CI −0.023 ถึง +0.080 → ยังสรุปไม่ได้ว่าต่างกันจริง; FP ไม่ลดลง (25 → 27) แปลว่า TF-IDF ไม่ได้ช่วยเรื่องคำหยาบที่ไม่ toxic
- M3 − B1 (seed 42): accuracy +0.176, McNemar p < 0.001, bootstrap 95% CI +0.097 ถึง +0.256; ทุก seed (42–46) p < 0.01 และ CI ไม่คร่อม 0
- FP rate บนทวีต non-toxic: B0 28% / B1 31% / M3 17% — FP ของ B1 27 ทวีต M3 ทายถูก 23 ทวีต แต่ M3 มี FP ใหม่ 11 ทวีต
- M3: lr = 5e-5, 5 epochs (เลือกจาก dev 10%), batch 16, max_len 128; รันบน MacBook Pro M5 (MPS) ~3.6 นาที/seed
- หมายเหตุ: ค่าที่เลือกอยู่ขอบ grid (lr สูงสุด, epoch สูงสุด) และ dev F1 ยังขึ้นอยู่ — อาจยังไม่ใช่ค่าที่ดีที่สุด
- ผลทั้งหมดหลังแก้การย่อคำยืดให้ย่อเฉพาะตัวอักษรซ้ำ ≥ 3 ตัว (เดิมย่อซ้ำ 2 ตัวด้วยทำให้คำปกติอย่าง "คนนั้น" เพี้ยน)

| ขั้นตอน | งาน | สถานะ |
|---|---|---|
| 1 | หัวข้อ, งานเดิม, gap, RQ, dataset protocol | ✅ เสร็จ |
| 2 | B0 และ B1 | ✅ เสร็จ |
| 3 | จัดโครงสร้างโค้ด + ใช้ split/seed ชุดเดียวกันทุกโมเดล | ✅ เสร็จ |
| 4 | Fine-tune M3 (WangchanBERTa) | ✅ เสร็จ |
| 5 | เปรียบเทียบผล, error analysis | ⏳ ถัดไป |
| 6 | Paper draft, mini conference, final package | ⏳ ถัดไป |

<p align="right"><a href="#top">⬆ กลับด้านบน</a></p>

---

<a id="code"></a>

## 💻 Code

<a id="c-structure"></a>

### โครงสร้างโฟลเดอร์

```
Code/
├── make_split.py          ขั้นที่ 1  สร้างชุดทดสอบ 176 ทวีต + ชุดฝึกสมดุล -> data/splits/
├── train_b0.py            ขั้นที่ 2  B0: นับคำ + LR (ทำซ้ำงานเดิม)       -> results/b0/
├── train_b1.py            ขั้นที่ 2  B1: TF-IDF + Logistic Regression     -> results/b1/
├── train_m3.py            ขั้นที่ 3  M3: WangchanBERTa fine-tune (GPU)     -> results/m3/
├── m3_colab.ipynb         โน้ตบุ๊กสำหรับรัน train_m3.py บน Google Colab
├── compare.py             ขั้นที่ 4  เทียบสองโมเดลทีละทวีต (McNemar + bootstrap)
├── compare_all.py         ขั้นที่ 4  เทียบทุกโมเดลพร้อมกัน (B0/B1/M3) + Holm + รูปแบบการทายผิด
├── src/                   โค้ดที่ทุกสคริปต์ใช้ร่วมกัน
│   ├── config.py          ค่าตั้งกลาง: path, seed, ขนาดชุดข้อมูล, เกณฑ์เลือกพารามิเตอร์
│   ├── data.py            โหลดข้อมูลดิบ, สร้าง/อ่านชุดฝึก-ชุดทดสอบ
│   ├── preprocess.py      ตัดคำ (Deepcut) และทำความสะอาดข้อความ (ใช้กับ B0/B1)
│   ├── linear.py          pipeline + grid search ที่ B0 และ B1 ใช้ร่วมกัน (ต่างกันแค่ vectorizer)
│   ├── transformer.py     M3: เตรียมข้อความสำหรับ WangchanBERTa, ฝึก/ทาย (PyTorch)
│   └── evaluate.py        เมตริก, บันทึกผล, เปรียบเทียบแบบจับคู่
├── data/
│   ├── raw/               thai_toxic_2160.csv, toxic_keywords.txt
│   └── splits/            train_balanced.csv (1,472), test_176.csv (176)
├── results/               ผลของแต่ละโมเดล (สร้างตอนรัน)
├── legacy/                โค้ด b0.py / b1.py และผลเดิมก่อนจัดโครงสร้าง (เก็บไว้อ้างอิง)
├── requirements.txt
└── requirements_m3.txt   ไลบรารีเพิ่มสำหรับ M3 (torch, transformers, sentencepiece)
```

<a id="c-run"></a>

### วิธีรัน

```bash
python3 -m venv venv_deepcut
source venv_deepcut/bin/activate
python -m pip install -r requirements.txt

python make_split.py      # มีไฟล์ใน data/splits/ อยู่แล้ว จะไม่เขียนทับ (ใช้ --force ถ้าจะสร้างใหม่)
python train_b0.py
python train_b1.py
python compare.py --a results/b0 --b results/b1
python compare.py --a results/b1 --b results/m3    # หลังมีผล M3
python compare_all.py                              # เทียบ B0/B1/M3 พร้อมกัน -> results/compare_all.json, compare_all_errors.csv
```

**M3 (ต้องใช้ GPU):** เปิด `m3_colab.ipynb` บน Google Colab (T4) แล้วรันตามลำดับ หรือบนเครื่องที่มี GPU:

```bash
pip install -r requirements_m3.txt
python train_m3.py --smoke   # ทดสอบว่าโค้ดรันได้ (~1–2 นาที, ผลไม่มีความหมาย)
python train_m3.py           # เลือก lr/epoch จาก dev 10% -> refit บนชุดฝึกทั้งหมด 5 seeds -> results/m3/
```

`results/m3/` = ผลของ seed 42 (ใช้กับ `compare.py`) และ `metrics.json` มี `seed_summary` (mean ± SD ทุก seed); ผลแยกแต่ละ seed อยู่ใน `results/m3/seeds/`

ทดสอบว่าโค้ดรันได้เร็ว ๆ โดยไม่ใช้ Deepcut: `A_SMOKE=1 python train_b1.py` (ตัวเลขผลไม่มีความหมาย)

<a id="c-output"></a>

### ไฟล์ผลลัพธ์ของแต่ละโมเดล (`results/<model>/`)

| ไฟล์ | เนื้อหา |
|---|---|
| `metrics.json` | precision/recall/F1 (toxic), accuracy, macro-F1, confusion matrix, พารามิเตอร์ที่เลือก, seed, เวอร์ชันไลบรารี |
| `predictions.csv` | `tweet_id, is_toxic, keyword, pred` ทุกทวีตในชุดทดสอบ — ใช้กับ `compare.py` |
| `errors.csv` | เฉพาะทวีตที่ทายผิด พร้อม `error_type` (FP/FN) — ใช้ทำ error analysis |

**ข้อตกลงสำหรับ M3:** อ่านข้อมูลด้วย `src.data.load_splits()` และบันทึกผลด้วย
`src.evaluate.save_run(out_dir, test, pred, metrics)` เพื่อให้ `compare.py` เทียบกับ B1 ได้ทันที

<a id="c-protocol"></a>

### Protocol (สรุป)

- ชุดทดสอบ: 44 คำสำคัญ × (2 toxic + 2 non-toxic) = 176 ทวีต จากทวีตที่ annotator เห็นตรงกัน 3-0
- ชุดฝึก: ทวีตที่เหลือ สมดุลต่อคำสำคัญ ไม่เกิน 30 ทวีตต่อป้าย = 736/736
- seed = 42, ตรวจแล้วว่าไม่มีทวีต (id และข้อความ) ซ้ำระหว่างชุดฝึกกับชุดทดสอบ
- B0/B1 เลือกพารามิเตอร์ด้วย 3-fold CV ภายในชุดฝึก (macro-F1) แล้วทดสอบบนชุดทดสอบครั้งเดียว

<p align="right"><a href="#top">⬆ กลับด้านบน</a></p>
