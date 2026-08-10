# -*- coding: utf-8 -*-
"""
ร้อยทุกขั้นตอนเข้าด้วยกัน — เป็นจุดเดียวที่ app.py เรียกใช้

ลำดับการทำงาน (ตรงกับ SCHEMA.md):
    raw_text
      -> cleansing   ลบ PII / HTML / อีโมจิ, แยกคำติดกัน   [กลุ่ม A]
      -> normalize   ยุบคำลากเสียง, แก้คำพิมพ์ผิด           [กลุ่ม A]
      -> tokenize    ตัดคำ, ลบ stopwords                    [กลุ่ม A]
      -> pos_tag     ติดป้ายชนิดคำ                          [กลุ่ม A]
      -> extract     สกัด entity ด้วย regex + NER           [กลุ่ม B]
      -> aspects     จัดกลุ่มหัวข้อ + วิเคราะห์ขั้วราย aspect [กลุ่ม C]
      -> screening   ประเมินความน่าเชื่อถือ                  [กลุ่ม D]
"""

from __future__ import annotations

import pandas as pd

from . import aspects as aspects_mod
from . import cleansing, extract, normalize, screening
from .config import ASPECTS

# คอลัมน์ที่ export ลง CSV (ตัดฟิลด์ที่เป็น dict/list ซ้อนออก)
EXPORT_COLUMNS = [
    "review_id", "lang", "review_status", "credibility_score", "overall_sentiment",
    "sentiment_score", "product_type", "brand", "seller_shop", "price", "delivery_days",
    "courier", "location", "aspects", "pros", "cons", "spam_signals", "clean_text",
]


def analyze(
    raw_text: str,
    review_id: str = "-",
    product_name: str = "",
    use_spell_checker: bool = False,
    duplicate_of: str | None = None,
) -> dict:
    """วิเคราะห์รีวิว 1 รายการ คืน dict ตาม schema ทั้ง 4 กลุ่ม"""
    raw_text = (raw_text or "").strip()

    # ---- กลุ่ม A: ทำความสะอาดและเตรียมข้อความ
    cleaned = cleansing.clean(raw_text)
    lang = cleaned["lang"]
    normalized = normalize.normalize(cleaned["clean_text"], lang)

    tokens_all = normalize.tokenize(normalized, lang, drop_stopwords=False)
    tokens_all = normalize.fix_typos(tokens_all, lang, use_spell_checker)
    tokens_clean = normalize.remove_stopwords(tokens_all, lang)
    tagged = normalize.pos_tag(tokens_all, lang)

    # ข้อความที่ตัดคำแล้วเชื่อมด้วยช่องว่าง — เป็นรูปแบบที่ matching.py ต้องการ
    # เพื่อกันไม่ให้คำค้นภาษาไทยไปแมตช์กลางคำ (เช่น "ทน" ใน "ตัวแทน")
    tokenized = " ".join(tokens_all)

    # ---- กลุ่ม B: สกัด entity
    entities = extract.extract_all(
        cleaned["clean_text"], normalized, tokenized, lang, tagged, product_name
    )

    # ---- กลุ่ม C: หัวข้อและความรู้สึก
    aspect_result = aspects_mod.analyze(tokenized, tagged)

    # ---- กลุ่ม D: คัดกรอง
    screen_result = screening.screen(
        raw_text=raw_text,
        clean_text=tokenized or cleaned["clean_text"],
        tokens=tokens_clean,
        pii_removed=cleaned["pii_removed"],
        aspects=aspect_result["aspects"],
        emoji_count=cleaned["emoji_count"],
        duplicate_of=duplicate_of,
    )

    keywords = list(dict.fromkeys(normalize.content_words(tagged)))

    return {
        "review_id": review_id,
        "raw_text": raw_text,
        **cleaned,
        "normalized_text": normalized,
        "tokens": tokens_all,
        "tokens_clean": tokens_clean,
        "pos_tags": tagged,
        "keywords": keywords[:10],
        **entities,
        **aspect_result,
        **screen_result,
        "fingerprint": screening.text_fingerprint(raw_text),
    }


def analyze_batch(
    texts: list[str],
    review_ids: list[str] | None = None,
    product_names: list[str] | None = None,
    progress=None,
) -> list[dict]:
    """
    วิเคราะห์หลายรีวิวพร้อมกัน

    ต่างจากการเรียก analyze() ทีละรายการตรงที่ตรวจ "รีวิวซ้ำ" ได้
    เพราะต้องเทียบข้ามรายการ จึงทำสองรอบ: รอบแรกหาลายนิ้วมือ รอบสองวิเคราะห์เต็ม
    """
    review_ids = review_ids or [f"R{i:04d}" for i in range(1, len(texts) + 1)]
    product_names = product_names or [""] * len(texts)

    # รอบที่ 1: จับคู่ข้อความซ้ำ โดยให้รายการแรกที่พบเป็นต้นฉบับ
    seen: dict[str, str] = {}
    duplicate_of: list[str | None] = []
    for text, rid in zip(texts, review_ids):
        fp = screening.text_fingerprint(text or "")
        duplicate_of.append(seen.get(fp))
        seen.setdefault(fp, rid)

    # รอบที่ 2: วิเคราะห์เต็มรูปแบบ
    results = []
    for i, (text, rid, pname) in enumerate(zip(texts, review_ids, product_names)):
        results.append(analyze(text, rid, pname, duplicate_of=duplicate_of[i]))
        if progress is not None:
            progress((i + 1) / len(texts))
    return results


def to_dataframe(results: list[dict]) -> pd.DataFrame:
    """แปลงผลลัพธ์เป็นตารางแบน พร้อมคอลัมน์ขั้วของแต่ละ aspect"""
    rows = []
    for r in results:
        row = {c: r.get(c) for c in EXPORT_COLUMNS}
        for key in ("brand", "seller_shop", "courier", "location", "aspects",
                    "pros", "cons", "spam_signals"):
            row[key] = ", ".join(map(str, r.get(key) or []))
        for key in ASPECTS:
            row[f"aspect_{key}"] = r["aspect_sentiment"].get(key, "")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(results: list[dict]) -> dict:
    """สรุปผลระดับชุดข้อมูล สำหรับหน้าวิเคราะห์หลายรีวิว"""
    passed = [r for r in results if r["review_status"] != "สแปม"]

    aspect_summary = []
    for key, spec in ASPECTS.items():
        mentions = [r["aspect_sentiment"].get(key) for r in passed if key in r["aspect_sentiment"]]
        if not mentions:
            continue
        n = len(mentions)
        aspect_summary.append({
            "aspect": f"{spec['icon']} {spec['label_th']}",
            "key": key,
            "mentions": n,
            "pos": mentions.count("pos"),
            "neg": mentions.count("neg"),
            "neutral": mentions.count("neutral"),
            "pos_pct": round(mentions.count("pos") / n * 100, 1),
            "neg_pct": round(mentions.count("neg") / n * 100, 1),
        })

    def top_terms(field: str, limit: int = 8) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for r in passed:
            for term in r.get(field) or []:
                counts[term] = counts.get(term, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]

    prices = [r["price"] for r in passed if r.get("price")]
    days = [r["delivery_days"] for r in passed if r.get("delivery_days")]

    return {
        "total": len(results),
        "passed": sum(1 for r in results if r["review_status"] == "ผ่าน"),
        "suspect": sum(1 for r in results if r["review_status"] == "น่าสงสัย"),
        "spam": sum(1 for r in results if r["review_status"] == "สแปม"),
        "aspect_summary": aspect_summary,
        "top_pros": top_terms("pros"),
        "top_cons": top_terms("cons"),
        "price_range": (min(prices), max(prices)) if prices else None,
        "avg_delivery_days": round(sum(days) / len(days), 1) if days else None,
        "top_brands": top_terms("brand", 6),
        "top_couriers": top_terms("courier", 5),
        "sentiment_counts": {
            s: sum(1 for r in passed if r["overall_sentiment"] == s)
            for s in ("positive", "mixed", "neutral", "negative")
        },
    }
