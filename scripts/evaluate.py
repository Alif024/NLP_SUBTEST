# -*- coding: utf-8 -*-
"""
ประเมินความแม่นยำของ pipeline เทียบกับ ground truth

ใช้สองแหล่ง:
  1. ชุดไทย   - เทียบ aspect กับคอลัมน์ expected_aspects ที่กำกับไว้เอง (แม่นที่สุด)
  2. ชุดอังกฤษ - เทียบ overall_sentiment กับคอลัมน์ Sentiment ของ Flipkart
                 (label ต้นทางมี noise จึงดูเป็นแนวโน้ม ไม่ใช่ค่าชี้ขาด)
  3. case_tags - ตรวจว่าเคสที่ตั้งใจทดสอบถูกจับได้จริงหรือไม่

รัน: python scripts/evaluate.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reviewlens import analyze_batch  # noqa: E402
from reviewlens.config import ASPECTS  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "reviews_sample.csv"


def parse_expected(value: str) -> dict[str, str]:
    """แปลง 'quality:pos|shipping:neg' เป็น dict"""
    out = {}
    for part in str(value or "").split("|"):
        if ":" in part:
            key, polarity = part.split(":", 1)
            out[key.strip()] = polarity.strip()
    return out


def main() -> None:
    df = pd.read_csv(DATA, keep_default_na=False)
    print(f"โหลด {len(df)} รีวิว\n")

    start = time.perf_counter()
    results = analyze_batch(
        df["raw_text"].tolist(), df["review_id"].tolist(), df["product_name"].tolist()
    )
    elapsed = time.perf_counter() - start
    print(f"ประมวลผลเสร็จใน {elapsed:.1f} วินาที ({elapsed / len(df) * 1000:.0f} ms/รีวิว)\n")

    by_id = {r["review_id"]: r for r in results}

    # ---------------------------------------------- 1. Aspect เทียบ ground truth ไทย
    print("=" * 62)
    print("1. ความแม่นยำของ Aspect (ชุดไทยที่กำกับ ground truth ไว้)")
    print("=" * 62)

    tp = fp = fn = 0
    polarity_correct = polarity_total = 0
    mistakes = []

    for _, row in df[df["lang"] == "th"].iterrows():
        expected = parse_expected(row["expected_aspects"])
        if not expected:
            continue
        got = by_id[row["review_id"]]["aspect_sentiment"]

        tp += len(set(expected) & set(got))
        fp += len(set(got) - set(expected))
        fn += len(set(expected) - set(got))

        for key in set(expected) & set(got):
            polarity_total += 1
            if expected[key] == got[key]:
                polarity_correct += 1
            else:
                mistakes.append(f"  {row['review_id']} {key}: คาด {expected[key]} ได้ {got[key]}")

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(f"  ตรวจพบ aspect  : precision {precision:.1%}  recall {recall:.1%}  F1 {f1:.1%}")
    print(f"  ความถูกของขั้ว : {polarity_correct}/{polarity_total} = "
          f"{polarity_correct / polarity_total:.1%}" if polarity_total else "  ไม่มีข้อมูล")
    if mistakes:
        print(f"\n  ขั้วที่ทายผิด ({len(mistakes)} รายการ):")
        print("\n".join(mistakes[:12]))

    # ---------------------------------------------- 2. Sentiment เทียบ Flipkart
    print("\n" + "=" * 62)
    print("2. Overall sentiment เทียบ label ของ Flipkart (ชุดอังกฤษ)")
    print("=" * 62)

    en = df[df["lang"] == "en"]
    matrix: dict[tuple[str, str], int] = {}
    for _, row in en.iterrows():
        got = by_id[row["review_id"]]["overall_sentiment"]
        matrix[(row["sentiment_label"], got)] = matrix.get((row["sentiment_label"], got), 0) + 1

    labels = ["positive", "neutral", "negative"]
    predictions = ["positive", "mixed", "neutral", "negative"]
    header = "  ต้นทาง \\ ทำนาย  " + "".join(f"{p:>10}" for p in predictions)
    print(header)
    agree = total = 0
    for label in labels:
        row_counts = [matrix.get((label, p), 0) for p in predictions]
        print(f"  {label:<16}" + "".join(f"{c:>10}" for c in row_counts))
        total += sum(row_counts)
        # นับว่าตรงกันเมื่อ label ตรง หรือเมื่อ "mixed" ถูกทายให้รีวิวที่ไม่ใช่กลาง
        agree += matrix.get((label, label), 0)
    print(f"\n  ตรงกันตรง ๆ {agree}/{total} = {agree / total:.1%}"
          f"  (label ต้นทางมี noise — ดูเป็นแนวโน้มเท่านั้น)")

    # ---------------------------------------------- 3. เคสที่ตั้งใจทดสอบ
    print("\n" + "=" * 62)
    print("3. เคสคัดกรองที่ตั้งใจทดสอบ")
    print("=" * 62)

    # เงื่อนไขคัดแถวที่ "ควรจับได้จริง" — บาง tag จาก prepare_data กว้างกว่าที่ระบบสกัด
    # เช่น tag "delivery_days" ครอบคลุม weeks/months ด้วย แต่ระบบดึงเฉพาะหน่วยวัน
    # และรีวิวต้นฉบับของข้อความซ้ำไม่ควรถูกนับว่าเป็นรีวิวซ้ำ
    checks = [
        ("ad_spam", None, lambda r: r["is_ad_spam"], "จับว่าเป็นโฆษณาแฝง"),
        ("duplicate", lambda t: True, lambda r: r["is_duplicate"], "จับว่าซ้ำ (ไม่นับต้นฉบับ)"),
        ("low_quality", None, lambda r: r["is_low_quality"], "จับว่าคุณภาพต่ำ"),
        ("pii_phone", None, lambda r: "phone" in r["pii_removed"], "ลบเบอร์โทรออก"),
        ("pii_line", None, lambda r: "line_id" in r["pii_removed"], "ลบ LINE ID ออก"),
        ("pii_url", None, lambda r: "url" in r["pii_removed"], "ลบ URL ออก"),
        ("pii_email", None, lambda r: "email" in r["pii_removed"], "ลบอีเมลออก"),
        ("elongation", None, lambda r: r["normalized_text"] != r["clean_text"], "ยุบคำลากเสียง"),
        ("delivery_days", lambda t: re.search(r"\d+\s*(วัน|days?)\b", t, re.I),
         lambda r: r["delivery_days"] is not None, "ดึงจำนวนวันจัดส่ง"),
        ("price", lambda t: re.search(r"(฿|rs\.?|₹)\s*\d|\d\s*(บาท|rs\b|rupees)", t, re.I),
         lambda r: r["price"] is not None, "ดึงราคา"),
        ("html", None, lambda r: r["had_html"], "ตรวจพบ HTML"),
        ("code_switch", None, lambda r: bool(r["aspects"]), "วิเคราะห์ข้อความปนภาษาได้"),
    ]

    duplicate_originals = {
        r["duplicate_of"] for r in results if r["duplicate_of"]
    }

    for tag, applicable, predicate, label in checks:
        subset = df[df["case_tags"].str.contains(tag, regex=False)]
        if applicable is not None:
            if tag == "duplicate":
                subset = subset[~subset["review_id"].isin(duplicate_originals)]
            else:
                subset = subset[subset["raw_text"].map(lambda t: bool(applicable(t)))]
        if subset.empty:
            continue
        hits = sum(1 for _, row in subset.iterrows() if predicate(by_id[row["review_id"]]))
        mark = "OK " if hits == len(subset) else "!! "
        print(f"  {mark}{label:<28} {hits}/{len(subset)}")

    # ---------------------------------------------- 4. ภาพรวมสถานะ
    print("\n" + "=" * 62)
    print("4. ผลการคัดกรองทั้งชุด")
    print("=" * 62)
    status = pd.Series([r["review_status"] for r in results]).value_counts()
    for name, count in status.items():
        print(f"  {name:<12} {count:>4}  ({count / len(results):.1%})")

    no_aspect = sum(1 for r in results if not r["aspects"])
    print(f"\n  รีวิวที่ไม่พบ aspect เลย: {no_aspect} ({no_aspect / len(results):.1%})")


if __name__ == "__main__":
    main()
