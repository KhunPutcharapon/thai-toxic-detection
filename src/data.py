"""โหลดข้อมูลดิบ และสร้าง/อ่านชุดทดสอบ-ชุดฝึกที่ทุกโมเดล (B0, B1, M3) ใช้ร่วมกัน"""
import numpy as np
import pandas as pd

from . import config


# โหลดข้อมูลดิบและคำสำคัญ 44 คำ (เพิ่มคอลัมน์ agree = จำนวนโหวตฝ่ายที่ชนะ: 3 = เห็นตรงกัน, 2 = เห็นต่าง)
def load_raw(path=config.RAW_CSV, kw_path=config.KEYWORDS_TXT):
    # อ่าน tweet_id เป็นข้อความ (กันเลขยาวถูกแปลงเป็นทศนิยม) และทิ้งแถวที่ไม่มีข้อความ
    df = pd.read_csv(path, dtype={"tweet_id": str}).dropna(subset=["tweet_text"]).reset_index(drop=True)
    df["agree"] = df[["toxic_votes", "nontoxic_votes"]].max(axis=1)
    with open(kw_path, encoding="utf-8") as f:
        kws = [line.strip() for line in f if line.strip()]
    assert len(kws) == config.N_KEYWORDS, f"คาดว่า {config.N_KEYWORDS} คำสำคัญ แต่ได้ {len(kws)}"
    return df, kws


# หาว่าทวีตไหนมีคำสำคัญแต่ละคำ -> {คำ: mask}
def _keyword_masks(texts, kws):
    return {k: texts.str.contains(k, regex=False) for k in kws}


# เรียงคำสำคัญจากคำที่มีทวีตน้อยสุด ให้คำหายากเลือกก่อนจะได้ไม่ถูกคำอื่นแย่งทวีต
def _rarest_first(kws, masks):
    return sorted(kws, key=lambda k: masks[k].sum())


# สร้างชุดทดสอบ 176 ทวีต: คำสำคัญละ 2 toxic + 2 non-toxic จากทวีตโหวต 3-0 -> คืน (test, ทวีตที่เหลือ)
def make_test(df, kws, pool=config.TEST_POOL, seed=config.SEED):
    cand = df if pool == "all" else df[df.agree == 3]
    has = _keyword_masks(cand.tweet_text, kws)
    rng = np.random.default_rng(seed)
    taken, rows = set(), []
    for k in _rarest_first(kws, has):
        for label in (1, 0):                                             # 1 = toxic, 0 = non-toxic
            ok = cand[has[k] & (cand.is_toxic == label) & (~cand.index.isin(taken))]
            if len(ok) < config.TEST_PER_LABEL:
                raise RuntimeError(f"คำ '{k}' มีทวีตคลาส {label} ไม่พอ ({len(ok)})")
            pick = rng.choice(ok.index.values, config.TEST_PER_LABEL, replace=False)
            taken.update(pick)
            rows += [(i, k) for i in pick]
    test = df.loc[[i for i, _ in rows]].copy()
    test["keyword"] = [k for _, k in rows]                               # คำสำคัญที่ทวีตถูกเลือกในนามของ
    rest = df.drop(index=[i for i, _ in rows])
    return test, rest


# สร้างชุดฝึกจากทวีตที่เหลือ: toxic = non-toxic ในแต่ละคำสำคัญ ไม่เกิน 30 ต่อป้าย -> คืน (train, จำนวนข้อความซ้ำกับชุดทดสอบ)
def make_train(rest, test, kws, seed=config.SEED):
    has = _keyword_masks(rest.tweet_text, kws)
    rng = np.random.default_rng(seed + 1)                                # seed คนละตัวกับตอนสุ่มชุดทดสอบ
    taken, kw_of = [], {}
    for k in _rarest_first(kws, has):
        pools = {lab: rest[has[k] & (rest.is_toxic == lab) & (~rest.index.isin(taken))] for lab in (1, 0)}
        n = min(config.MAX_TRAIN_PER_LABEL, len(pools[1]), len(pools[0]))
        for lab in (1, 0):
            pick = rng.choice(pools[lab].index.values, n, replace=False)
            taken += list(pick)
            kw_of.update({i: k for i in pick})
    train = rest.loc[taken].copy()
    train["keyword"] = [kw_of[i] for i in taken]
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)   # สลับลำดับแถว
    assert train.is_toxic.sum() * 2 == len(train), "ชุดฝึกไม่สมดุล"
    assert set(train.tweet_id).isdisjoint(test.tweet_id), "ชุดฝึกกับชุดทดสอบมีทวีตซ้ำ!"
    # ข้อความที่เหมือนกันทุกตัวอักษร (แม้คนละ id เช่นรีทวีต) ก็นับเป็นการรั่วได้
    n_dup_text = len(set(train.tweet_text.str.strip()) & set(test.tweet_text.str.strip()))
    return train, n_dup_text


# บันทึกชุดฝึก/ชุดทดสอบลง data/splits/
def save_splits(train, test, train_path=config.TRAIN_CSV, test_path=config.TEST_CSV):
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test[config.SPLIT_COLUMNS].to_csv(test_path, index=False, encoding="utf-8-sig")
    train[config.SPLIT_COLUMNS].to_csv(train_path, index=False, encoding="utf-8-sig")


# อ่านชุดฝึก/ชุดทดสอบจาก data/splits/ (ทุกโมเดลต้องอ่านผ่านฟังก์ชันนี้ เพื่อใช้ชุดเดียวกัน)
def load_splits(train_path=config.TRAIN_CSV, test_path=config.TEST_CSV):
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"ไม่พบ {p} — รัน `python make_split.py` ก่อน")
    train = pd.read_csv(train_path, dtype={"tweet_id": str})
    test = pd.read_csv(test_path, dtype={"tweet_id": str})
    return train, test
