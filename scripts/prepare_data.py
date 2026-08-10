# -*- coding: utf-8 -*-
"""
เตรียมไฟล์ข้อมูลทดสอบสำหรับ ReviewLens TH

รันครั้งเดียวตอนเตรียมโปรเจกต์ (ไม่ได้ถูกเรียกตอน deploy):
    python scripts/prepare_data.py

ผลลัพธ์:
    data/reviews_flipkart_en.csv  - สุ่มจาก Flipkart 205k แถว แบบ stratified
    data/reviews_th.csv           - ชุดรีวิวไทยที่เขียนเอง (จาก scripts/thai_seed.py)
    data/reviews_sample.csv       - รวมสองชุด = ไฟล์ที่แอปโหลดจริง

หมายเหตุเรื่อง dependency:
    kagglehub ใช้เฉพาะตอนเตรียมข้อมูล จึงอยู่ใน requirements-dev.txt เท่านั้น
    ไม่ต้องใส่ใน requirements.txt ของ Streamlit เพราะ CSV ถูก commit ลงรีโปไปแล้ว
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from thai_seed import THAI_REVIEWS  # noqa: E402

KAGGLE_DATASET = "niraliivaghani/flipkart-product-customer-reviews-dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TARGET_EN_ROWS = 500
RANDOM_STATE = 42

# คอลัมน์กลางที่ทั้งสองภาษาใช้ร่วมกัน
UNIFIED_COLUMNS = [
    "review_id",
    "lang",
    "source",
    "raw_text",
    "product_name",
    "product_price",
    "rate",
    "sentiment_label",
    "case_tags",
    "expected_aspects",
]

# ---------------------------------------------------------------- ตัวช่วยตรวจเคส

# Flipkart ลบเครื่องหมายวรรคตอนออกหมดแล้ว จึงไม่มี URL/เบอร์โทรให้ตรวจ
# แพตเทิร์นด้านล่างใช้ "ติดป้ายเคส" ให้แถวที่สุ่มได้ ไม่ใช่การทำความสะอาด
PATTERNS = {
    "delivery_days": re.compile(r"\b\d+\s*(?:days?|weeks?|months?)\b", re.I),
    "price": re.compile(r"(?:rs\.?|₹|rupees?|price)\s*\.?\s*\d|worth\s+(?:the|of)\s+money", re.I),
    "elongation": re.compile(r"([a-z])\1{2,}", re.I),
    "packaging": re.compile(r"\b(?:packag\w*|box|damaged|broken|sealed|wrapp\w*)\b", re.I),
    "shipping": re.compile(r"\b(?:deliver\w*|shipp?\w*|courier|dispatch\w*|late|on time)\b", re.I),
    "quality": re.compile(r"\b(?:qualit\w*|durab\w*|build|material|defect\w*|fake|original)\b", re.I),
    "seller_service": re.compile(r"\b(?:seller|service|support|replace\w*|refund\w*|return\w*|complain\w*)\b", re.I),
    "as_described": re.compile(r"\b(?:as (?:described|shown|per)|different|wrong (?:item|size|colou?r)|not same|mismatch\w*)\b", re.I),
}

MOJIBAKE = re.compile(r"(?:\?{3,}|\?ÃÂ¿|Ã.|Â.)")


def tag_cases(text: str) -> list[str]:
    """ติดป้ายว่าแถวนี้ใช้ทดสอบเคสอะไรได้บ้าง"""
    tags = [name for name, pat in PATTERNS.items() if pat.search(text)]
    if len(text.split()) <= 3:
        tags.append("low_quality")
    if len(text) >= 200:
        tags.append("long")
    if len(tags) >= 4:
        tags.append("multi_aspect")
    return tags


def clean_product_name(name: str) -> str:
    """ลบ mojibake ที่ติดมากับชื่อสินค้าใน dataset ต้นฉบับ"""
    name = MOJIBAKE.sub(" ", str(name))
    return re.sub(r"\s{2,}", " ", name).strip()


# ---------------------------------------------------------------- ขั้นตอนหลัก


def download_flipkart() -> Path:
    import kagglehub

    print(f"[1/5] ดาวน์โหลด dataset: {KAGGLE_DATASET}")
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = list(path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ไม่พบไฟล์ CSV ใน {path}")
    print(f"      -> {csv_files[0]}")
    return csv_files[0]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    print("[2/5] โหลดและทำความสะอาด")
    df = pd.read_csv(csv_path)
    total = len(df)

    # dataset มีแถวเสียที่คอลัมน์เลื่อน สังเกตได้จาก Rate ที่ไม่ใช่ตัวเลข 1-5
    df = df[df["Rate"].astype(str).str.fullmatch(r"[1-5]")]
    print(f"      ตัดแถวคอลัมน์เลื่อนออก {total - len(df)} แถว")

    df = df.dropna(subset=["Summary"]).copy()
    df["Summary"] = df["Summary"].astype(str).str.strip()
    df = df[df["Summary"].str.len() > 0]

    df["rate"] = df["Rate"].astype(int)
    df["product_name"] = df["product_name"].map(clean_product_name)
    df["product_price"] = df["product_price"].astype(str).str.replace(r"[^\d.]", "", regex=True)

    print(f"      เหลือ {len(df):,} แถวที่ใช้ได้")
    return df


def sample_stratified(df: pd.DataFrame) -> pd.DataFrame:
    """
    สุ่มแบบมีโควตา เพื่อรับประกันว่าชุดทดสอบครอบคลุมทุกเคสที่แอปต้องรับมือ
    ไม่ใช้การสุ่มธรรมดา เพราะข้อมูลจริงเอียงไปทาง positive ถึง 81%
    และรีวิวส่วนใหญ่สั้นเกินกว่าจะมี aspect ให้วิเคราะห์ (median = 17 ตัวอักษร)
    """
    print(f"[3/5] สุ่มแบบ stratified ให้ได้ ~{TARGET_EN_ROWS} แถว")

    work = df.copy()
    work["_len"] = work["Summary"].str.len()
    work["_tags"] = work["Summary"].map(lambda t: "|".join(tag_cases(t)))

    picked: list[pd.DataFrame] = []
    used: set[int] = set()

    def balanced(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        สุ่มให้ทั้งสาม sentiment มีสัดส่วนใกล้เคียงกัน
        ถ้าคลาสไหนมีไม่พอ จะเติมจากคลาสที่เหลือแทน
        """
        per_class = -(-n // 3)  # ปัดขึ้น
        chunks, short = [], 0
        for sentiment in ("negative", "neutral", "positive"):
            sub = pool[pool["Sentiment"].eq(sentiment)]
            grab = min(per_class, len(sub))
            short += per_class - grab
            if grab:
                chunks.append(sub.sample(n=grab, random_state=RANDOM_STATE))
        if short:
            taken = pd.concat(chunks).index if chunks else []
            rest = pool[~pool.index.isin(taken)]
            if len(rest):
                chunks.append(rest.sample(n=min(short, len(rest)), random_state=RANDOM_STATE))
        return pd.concat(chunks).head(n) if chunks else pool.head(0)

    def take(mask: pd.Series, n: int, label: str, min_len: int = 0) -> None:
        pool = work[mask & ~work.index.isin(used)]
        if min_len:
            # ใช้ความยาวขั้นต่ำแทนการหยิบรีวิวที่ยาวที่สุด
            # เพื่อให้ยังได้ความหลากหลายของความยาว ไม่กระจุกที่เพดาน 497 ตัวอักษร
            pool = pool[pool["_len"] >= min_len]
        chunk = balanced(pool, n)
        used.update(chunk.index)
        picked.append(chunk)
        print(f"      {label:<28} {len(chunk):>4} แถว")

    has = lambda key: work["_tags"].str.contains(key, regex=False)  # noqa: E731

    # โควตาตามเคสที่ต้องทดสอบ (เรียงจากเคสที่หายากที่สุดก่อน)
    take(has("as_described"), 40, "ของไม่ตรงปก", min_len=80)
    take(has("packaging"), 45, "บรรจุภัณฑ์", min_len=80)
    take(has("delivery_days"), 55, "ระบุจำนวนวันจัดส่ง", min_len=60)
    take(has("price"), 40, "ระบุราคา/ความคุ้มค่า", min_len=60)
    take(has("seller_service"), 40, "บริการร้านค้า", min_len=80)
    take(has("elongation"), 35, "คำลากเสียง")
    take(has("multi_aspect"), 60, "หลาย aspect ในรีวิวเดียว", min_len=150)
    take(has("low_quality"), 45, "รีวิวสั้น/คุณภาพต่ำ")

    # เติมส่วนที่เหลือด้วยรีวิวความยาวปานกลาง เพื่อถ่วงไม่ให้ชุดข้อมูลมีแต่สั้นจัดกับยาวจัด
    remaining = TARGET_EN_ROWS - sum(len(p) for p in picked)
    if remaining > 0:
        take(work["_len"].between(30, 150), remaining, "เติม: ความยาวปานกลาง")

    out = pd.concat(picked).sample(frac=1, random_state=RANDOM_STATE)
    print(f"      รวม {len(out)} แถว")
    return out


def to_unified_en(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "review_id": [f"EN{i:04d}" for i in range(1, len(df) + 1)],
        "lang": "en",
        "source": "flipkart",
        "raw_text": df["Summary"].values,
        "product_name": df["product_name"].values,
        "product_price": df["product_price"].values,
        "rate": df["rate"].values,
        "sentiment_label": df["Sentiment"].values,
        "case_tags": df["_tags"].values,
        "expected_aspects": "",
    })
    # แถวที่ไม่เข้าเคสใดเลยต้องมีค่า ไม่ใช่ค่าว่าง เพื่อไม่ให้กลายเป็น NaN ตอนอ่านกลับ
    out["case_tags"] = out["case_tags"].replace("", "untagged")
    return out[UNIFIED_COLUMNS]


def to_unified_th() -> pd.DataFrame:
    df = pd.DataFrame(THAI_REVIEWS)
    df.insert(0, "review_id", [f"TH{i:04d}" for i in range(1, len(df) + 1)])
    df.insert(1, "lang", "th")
    df.insert(2, "source", "handcrafted")
    return df[UNIFIED_COLUMNS]


def build_en_vocab(df: pd.DataFrame, min_freq: int = 50) -> pd.Series:
    """
    สร้างคลังคำอังกฤษจาก corpus เต็ม 205k แถว

    ใช้แก้ปัญหาเฉพาะของ dataset นี้: เครื่องหมายวรรคตอนถูกลบไปหมด
    ทำให้คำติดกันตรงรอยต่อประโยค เช่น "productfeel", "acbut", "providedifficult"
    คำที่ปรากฏบ่อย (>= min_freq) ถือว่าเป็นคำจริง ส่วนคำที่ติดกันจะปรากฏน้อยมาก
    จึงใช้คลังนี้ตัดคำติดกันแบบอนุรักษ์นิยมได้ (ดู reviewlens/cleansing.py)
    """
    words = df["Summary"].str.lower().str.findall(r"[a-z]+").explode()
    freq = words.value_counts()
    vocab = freq[(freq >= min_freq) & (freq.index.str.len() >= 2)]
    print(f"      คลังคำอังกฤษ: {len(vocab):,} คำ (จาก {len(freq):,} คำที่พบทั้งหมด)")
    return vocab


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 58)
    print("สรุปชุดข้อมูลทดสอบ")
    print("=" * 58)
    print(f"รวมทั้งหมด {len(df)} แถว")
    print("\nแยกตามภาษา:")
    print(df["lang"].value_counts().to_string())
    print("\nแยกตาม sentiment:")
    print(df["sentiment_label"].value_counts().to_string())

    print("\nความครอบคลุมของเคสทดสอบ:")
    tags = df["case_tags"].fillna("").str.split("|").explode()
    tags = tags[tags.str.len() > 0]
    for tag, n in tags.value_counts().items():
        print(f"  {tag:<20} {n:>4}")

    lengths = df["raw_text"].str.len()
    print(f"\nความยาวข้อความ: min={lengths.min()} median={int(lengths.median())} max={lengths.max()}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    full = load_and_clean(download_flipkart())
    vocab = build_en_vocab(full)
    en = to_unified_en(sample_stratified(full))
    th = to_unified_th()

    print("\n[4/5] เขียนไฟล์")
    # เขียนช่องว่างเป็นสตริงว่าง ไม่ใช่ NaN เพื่อให้แอปอ่านกลับมาได้โดยไม่ต้องเช็ค null
    en, th = en.fillna(""), th.fillna("")
    for name, frame in [("reviews_flipkart_en.csv", en), ("reviews_th.csv", th)]:
        path = DATA_DIR / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"      {path.relative_to(PROJECT_ROOT)}  ({len(frame)} แถว, {path.stat().st_size / 1024:.0f} KB)")

    combined = pd.concat([th, en], ignore_index=True)
    combined_path = DATA_DIR / "reviews_sample.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"      {combined_path.relative_to(PROJECT_ROOT)}  ({len(combined)} แถว, "
          f"{combined_path.stat().st_size / 1024:.0f} KB)")

    vocab_path = DATA_DIR / "en_vocab.txt"
    vocab_path.write_text("\n".join(vocab.index), encoding="utf-8")
    print(f"      {vocab_path.relative_to(PROJECT_ROOT)}  ({len(vocab)} คำ, "
          f"{vocab_path.stat().st_size / 1024:.0f} KB)")

    print("\n[5/5] ตรวจผล")
    report(combined)


if __name__ == "__main__":
    main()
