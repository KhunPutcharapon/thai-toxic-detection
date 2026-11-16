"""ค่าตั้งกลางของทั้งโปรเจกต์ — ทุกสคริปต์อ่านค่าจากไฟล์นี้ที่เดียว
ถ้าจะเปลี่ยน seed / path / เกณฑ์ ให้แก้ที่นี่ แล้วบันทึกการเปลี่ยนในรายงาน
"""
from pathlib import Path

# ----- ตำแหน่งไฟล์ -----
ROOT = Path(__file__).resolve().parent.parent      # โฟลเดอร์ Code/
DATA_RAW = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"

RAW_CSV = DATA_RAW / "thai_toxic_2160.csv"         # ทวีตที่ยังดึงได้ 2,160 ทวีต พร้อมโหวตของ annotator 3 คน
KEYWORDS_TXT = DATA_RAW / "toxic_keywords.txt"     # คำสำคัญ 44 คำของงานเดิม (บรรทัดละคำ)
TRAIN_CSV = SPLIT_DIR / "train_balanced.csv"
TEST_CSV = SPLIT_DIR / "test_176.csv"

# ----- การสร้างชุดข้อมูล (เลียนแบบ Sirihattasak et al., 2018) -----
SEED = 42                  # เลขสุ่มคงที่ => ได้ชุดทดสอบ/ชุดฝึกชุดเดิมทุกครั้ง
N_KEYWORDS = 44
TEST_POOL = "unanimous"    # ชุดทดสอบเลือกจากทวีตที่โหวต 3-0 เท่านั้น ("all" = ทุกทวีต)
TEST_PER_LABEL = 2         # ชุดทดสอบ: คำสำคัญละ 2 toxic + 2 non-toxic => 44 x 4 = 176 ทวีต
MAX_TRAIN_PER_LABEL = 30   # ชุดฝึก: ไม่เกิน 30 ทวีตต่อป้ายต่อคำสำคัญ (งานเดิมระบุ)

# ----- การเลือกพารามิเตอร์ -----
SCORING = "f1_macro"       # เคยลอง "f1" ของคลาส toxic แล้วโมเดลเอียงไปทาย toxic เกือบทุกทวีต
INNER_FOLDS = 3            # CV ภายในชุดฝึก (ไม่แตะชุดทดสอบ)

# ----- คอลัมน์มาตรฐานของไฟล์ผลลัพธ์ -----
SPLIT_COLUMNS = ["tweet_id", "tweet_text", "toxic_votes", "nontoxic_votes", "is_toxic", "keyword"]

# ----- M3: WangchanBERTa fine-tune -----
M3_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"
M3_MAX_LEN = 128           # ทวีตยาวไม่เกิน ~142 ตัวอักษร => 128 subword token พอ (ไม่ถูกตัดทิ้ง)
M3_BATCH = 16
M3_LR_GRID = [2e-5, 3e-5, 5e-5]   # ค่ามาตรฐานของการ fine-tune BERT-family
M3_MAX_EPOCHS = 5          # ตอนเลือกพารามิเตอร์ วัด dev ทุก epoch แล้วเลือก epoch ที่ดีสุด
M3_WEIGHT_DECAY = 0.01
M3_WARMUP = 0.1            # 10% แรกของ step ค่อย ๆ เพิ่ม learning rate
M3_DEV_FRAC = 0.1          # แบ่ง 10% ของชุดฝึกเป็น dev ไว้เลือก lr/epoch เท่านั้น (ไม่แตะชุดทดสอบ)
M3_SEEDS = [42, 43, 44, 45, 46]   # ฝึกซ้ำหลาย seed แล้วรายงาน mean ± SD (ชุดทดสอบเล็ก ผลแกว่งตาม seed)
