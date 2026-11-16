"""ตัดคำและทำความสะอาดข้อความ สำหรับโมเดลแบบนับคำ/TF-IDF (B0, B1)
M3 (WangchanBERTa) ใช้ tokenizer ของโมเดลเอง ไม่ต้องใช้ไฟล์นี้"""
import os
import re

URL = re.compile(r"https?://\S+")    # ลิงก์
USER = re.compile(r"@\w+")           # @ผู้ใช้
THAI = re.compile(r"[฀-๿]")   # ตัวอักษรไทย


# คืน (ตัวตัดคำ, คำหยุด, ชื่อตัวตัดคำ): ใช้ Deepcut ตามงานเดิม ถ้าใช้ไม่ได้ถอยไป newmm | A_SMOKE=1 = ตัดตามช่องว่างไว้ทดสอบโค้ด
def get_tokenizer():
    if os.environ.get("A_SMOKE") == "1":
        return (lambda t: t.split()), set(), "space(smoke-test)"
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus.common import thai_stopwords
    stop = set(thai_stopwords())
    engine = "deepcut"
    try:
        word_tokenize("ทดสอบ", engine="deepcut")                         # ลองเรียกจริง กันกรณีติดตั้งแล้วแต่รันไม่ได้
    except Exception:
        engine = "newmm"
        print("⚠ ใช้ Deepcut ไม่ได้ -> ใช้ newmm แทน (ต้องบันทึกเป็นความต่างจากงานเดิมในรายงาน)")
    return (lambda t: word_tokenize(t, engine=engine, keep_whitespace=False)), stop, engine


# ย่อคำยืด เช่น มากกก -> มาก, 55555 -> 555 (ตามงานเดิม)
def squeeze_repeats(text):
    text = re.sub(r"5{3,}", "555", text)                                 # 55555 -> 555
    # ย่อเฉพาะตัวที่ซ้ำ >= 3 ตัว: ตัวซ้ำ 2 ตัวมักเป็นคำปกติ (คนนั้น, แบบ, ปัญญา) ไม่ใช่คำยืด
    return re.sub(r"([ก-๎])\1{2,}", r"\1", text)             # มากกก -> มาก


# ทำความสะอาดทวีต 1 ข้อความ: ลบลิงก์/@ผู้ใช้ -> ย่อคำยืด -> ตัดคำ -> ทิ้งคำที่ไม่ใช่ไทยและคำหยุด (เก็บ ? กับ !)
def preprocess(text, tok, stop):
    text = squeeze_repeats(USER.sub(" ", URL.sub(" ", text)))            # ลบลิงก์, @ผู้ใช้, ย่อตัวอักษรซ้ำ
    out = []
    for w in tok(text):
        w = w.strip()
        if not w:
            continue
        if w in ("?", "!"):                                              # งานเดิมเก็บ ? กับ ! ไว้
            out.append(w)
            continue
        if not THAI.search(w) or w in stop:                              # ทิ้งคำที่ไม่ใช่ไทย และคำหยุด
            continue
        out.append(w)
    return " ".join(out)


# ใช้ preprocess กับทั้งคอลัมน์
def preprocess_texts(texts, tok, stop):
    return texts.astype(str).apply(lambda t: preprocess(t, tok, stop))
