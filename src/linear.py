"""โมเดลเชิงเส้น B0 และ B1 — ต่างกันแค่ "วิธีแปลงข้อความเป็นตัวเลข" ส่วนอื่นใช้ร่วมกันที่นี่
B0: CountVectorizer (นับคำ แบบงานเดิม)   B1: TfidfVectorizer (ถ่วงน้ำหนัก IDF + ปรับความยาวเวกเตอร์เป็น 1)
สิ่งที่คงไว้เหมือนกัน: ชุดข้อมูล, การเตรียมข้อความ, ตัวตัดคำ, min_df, Logistic Regression, grid, เกณฑ์เลือก, seed"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config
from .data import load_splits
from .evaluate import compute_metrics, environment_info
from .preprocess import get_tokenizer, preprocess_texts

# C = ความแรงของ regularization (เล็ก = เรียบง่าย) | รวม 5 x 2 x 2 = 20 แบบ
PARAM_GRID = {"clf__C": [0.001, 0.01, 0.1, 1, 10],
              "clf__penalty": ["l1", "l2"],
              "clf__fit_intercept": [True, False]}


# สร้าง Pipeline (vectorizer -> Logistic Regression) + GridSearchCV; vectorizer_cls = CountVectorizer (B0) หรือ TfidfVectorizer (B1)
def build_search(vectorizer_cls):
    pipe = Pipeline([
        # min_df=10: เก็บคำที่พบ >= 10 ทวีต | token_pattern=\S+: คำที่ตัดแล้วคั่นด้วยช่องว่าง
        # (ค่าเริ่มต้นของ sklearn ตัดภาษาไทยพัง) | lowercase=False: ไม่แปลงตัวพิมพ์
        ("vec", vectorizer_cls(min_df=10, token_pattern=r"\S+", lowercase=False)),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000)),    # liblinear รองรับทั้ง l1 และ l2
    ])
    cv = StratifiedKFold(config.INNER_FOLDS, shuffle=True, random_state=config.SEED)
    return GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring=config.SCORING, n_jobs=-1)


# ดึงคะแนน CV ภายใน 5 อันดับแรก ไว้ดูว่าพารามิเตอร์ต่างกันมากไหม
def top_cv_results(gs, n=5):
    cv = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score"]].sort_values("mean_test_score", ascending=False)
    return [(str(r.params), round(float(r.mean_test_score), 4)) for r in cv.head(n).itertuples()]


# ทำงานครบทั้งสายของ B0/B1: อ่านข้อมูล -> ตัดคำ -> เลือกพารามิเตอร์ใน train -> ทาย test -> คืน (metrics, test, pred)
def run_linear_model(vectorizer_cls, model_name):
    train, test = load_splits()
    print(f"[ข้อมูล] ชุดฝึก {len(train)} ทวีต / ชุดทดสอบ {len(test)} ทวีต")

    tok, stop, engine = get_tokenizer()
    print(f"[ตัวตัดคำ] {engine}")
    train["text"] = preprocess_texts(train["tweet_text"], tok, stop)
    test["text"] = preprocess_texts(test["tweet_text"], tok, stop)

    gs = build_search(vectorizer_cls)
    gs.fit(train["text"], train["is_toxic"].values)                      # เลือกพารามิเตอร์เฉพาะในชุดฝึก
    pred = gs.predict(test["text"])                                      # ทดสอบกับชุดทดสอบครั้งเดียว

    m = compute_metrics(test["is_toxic"].values, pred)
    m.update(model=model_name,
             best_params={k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
             n_features=len(gs.best_estimator_.named_steps["vec"].vocabulary_),
             tokenizer=engine, top5_inner_cv=top_cv_results(gs),
             seed=config.SEED, scoring=config.SCORING, n_train=len(train), n_test=len(test),
             **environment_info())

    print(f"[พารามิเตอร์ที่เลือก] {m['best_params']}  (จำนวนฟีเจอร์ {m['n_features']})")
    for params, score in m["top5_inner_cv"]:
        print(f"  CV {score:.4f}  {params}")
    return m, test, pred
