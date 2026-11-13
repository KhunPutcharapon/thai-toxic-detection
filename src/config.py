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
