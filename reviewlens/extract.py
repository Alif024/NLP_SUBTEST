# -*- coding: utf-8 -*-
"""
ขั้นตอนที่ 3 ของ pipeline: POS & NER + Regex Extraction

ใช้สองวิธีร่วมกันโดยตั้งใจ เพราะแต่ละวิธีเก่งคนละอย่าง:
  - Regex  แม่นกับข้อมูลที่มีรูปแบบตายตัว (ราคา จำนวนวัน ดาว สี/ไซส์)
  - NER    เก่งกับชื่อเฉพาะที่เขียนได้หลายแบบ (แบรนด์ ร้าน สถานที่)

ตัวอย่างที่พิสูจน์ว่าต้องใช้คู่กัน: spaCy มอง "6195" ในประโยค
"the inverter costs Rs. 6195" เป็น DATE ไม่ใช่ MONEY แต่ regex ราคาจับได้ถูก
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import config as cfg
from .matching import find_hits, matched_terms
from .normalize import spacy_model

THAI_MONTHS = (
    "ม.ค.|ก.พ.|มี.ค.|เม.ย.|พ.ค.|มิ.ย.|ก.ค.|ส.ค.|ก.ย.|ต.ค.|พ.ย.|ธ.ค."
    "|มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน"
    "|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม"
)
THAI_DATE = re.compile(rf"\d{{1,2}}\s*(?:{THAI_MONTHS})\s*\d{{2,4}}|(?:{THAI_MONTHS})\s*\d{{2,4}}")


# ชื่อจังหวัดที่ซ้ำกับคำทั่วไป ต้องมีคำนำหน้าบอกบริบทถึงจะนับว่าเป็นสถานที่
# ("ได้เลยค่ะ" ไม่ใช่จังหวัดเลย, "ตากผ้า" ไม่ใช่จังหวัดตาก)
AMBIGUOUS_PROVINCES = {"เลย", "ตาก", "ตราด", "น่าน", "แพร่", "ระนอง"}
PROVINCE_MARKER = re.compile(r"(?:จังหวัด|จ\.|ถึง|ส่งไป|อยู่ที่|ที่)\s*$")


@lru_cache(maxsize=1)
def thai_provinces() -> list[str]:
    """รายชื่อ 77 จังหวัดจาก PyThaiNLP ใช้เป็น gazetteer แทนโมเดล NER ไทย"""
    try:
        from pythainlp.corpus import provinces

        return sorted(provinces(), key=len, reverse=True)
    except Exception:
        return []


def _thai_locations(text: str) -> list[str]:
    """หาชื่อจังหวัด พร้อมกรองชื่อที่พ้องกับคำทั่วไปออก"""
    out = []
    for name, start, _ in find_hits(text, thai_provinces()):
        if name in AMBIGUOUS_PROVINCES and not PROVINCE_MARKER.search(text[:start]):
            continue
        out.append(name)
    return out


# ---------------------------------------------------------------- Regex extraction


def extract_price(text: str) -> float | None:
    """
    ดึงราคาแรกที่พบ รองรับทั้ง ฿1,290 / 890 บาท / Rs. 6195 / 6195 rs
    """
    m = cfg.PRICE_PATTERN.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        return float(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def extract_delivery_days(text: str) -> int | None:
    """
    ดึงจำนวนวันจัดส่ง — เอาค่าน้อยที่สุดที่พบ

    เหตุผลที่เลือกค่าน้อยสุด: รีวิวมักพูดถึงหลายช่วงเวลาในประโยคเดียว
    เช่น "ใช้มา 30 วันแล้ว ส่งถึงใน 3 วัน" ตัวเลขที่เกี่ยวกับการจัดส่งมักเป็นตัวที่น้อยกว่า
    และกรองค่าเกิน 60 วันทิ้ง เพราะมักเป็นระยะเวลาการใช้งาน ไม่ใช่การจัดส่ง
    """
    days = [int(n) for n, _ in cfg.DELIVERY_DAYS_PATTERN.findall(text)]
    days = [d for d in days if 0 < d <= 60]
    return min(days) if days else None


def extract_star(text: str) -> float | None:
    m = cfg.STAR_PATTERN.search(text)
    if not m:
        return None
    value = float(m.group(1))
    return value if 0 <= value <= 5 else None


def extract_variant(text: str) -> dict[str, str]:
    """ดึงสี / ไซส์ / รุ่น ที่เขียนในรูปแบบ 'สี: ดำ' หรือ 'size L'"""
    out: dict[str, str] = {}
    for m in cfg.VARIANT_PATTERN.finditer(text):
        key = m.group(0).split()[0].rstrip(":：").strip().lower()
        value = m.group(1).strip()
        if value and value.lower() not in {"ที่", "และ", "the", "of"}:
            out.setdefault(key, value)
    return out


def extract_dates(text: str, lang: str) -> list[str]:
    """วันที่ในข้อความ — ไทยใช้ regex, อังกฤษใช้ NER ของ spaCy"""
    dates = [m.group(0).strip() for m in THAI_DATE.finditer(text)]
    if lang == "en":
        nlp = spacy_model()
        if nlp is not None:
            dates += [
                e.text
                for e in nlp(text).ents
                # ตัดพวก "3 days" ออก เพราะถูกจับเป็น delivery_days ไปแล้ว
                if e.label_ == "DATE" and not re.fullmatch(r"[\d\s]+(days?|weeks?|months?)", e.text, re.I)
            ]
    return sorted(set(dates))


# ---------------------------------------------------------------- Gazetteer + NER


def extract_named_entities(text: str, tokenized: str, lang: str,
                           tagged: list[tuple[str, str]]) -> dict:
    """
    สกัดชื่อเฉพาะ โดยรวมผลจาก 3 แหล่ง:
      1. gazetteer  - แบรนด์ / ขนส่ง / แพลตฟอร์ม / จังหวัด (แม่นที่สุด ควบคุมได้)
      2. NER model  - spaCy ORG/GPE สำหรับอังกฤษ (จับชื่อที่ไม่มีในคลัง)
      3. POS PROPN  - คำนามเฉพาะที่เหลือ ใช้เป็นตัวเลือกสำรอง

    text      : ข้อความที่ทำความสะอาดแล้ว ใช้กับ spaCy NER ที่ต้องการประโยคจริง
    tokenized : ข้อความที่ตัดคำแล้ว ใช้กับ gazetteer ภาษาไทยเพื่อกันการแมตช์กลางคำ
    """
    brands = matched_terms(text, cfg.BRANDS)
    couriers = matched_terms(text, cfg.COURIERS)
    platforms = matched_terms(text, cfg.PLATFORMS)
    locations = _thai_locations(tokenized) if lang == "th" else []

    ner_orgs: list[str] = []
    if lang == "en":
        nlp = spacy_model()
        if nlp is not None:
            for ent in nlp(text).ents:
                if ent.label_ == "ORG":
                    ner_orgs.append(ent.text)
                elif ent.label_ in ("GPE", "LOC"):
                    locations.append(ent.text)

    # ชื่อที่ NER เจอแต่ไม่มีในคลังแบรนด์/ขนส่ง ถือเป็นชื่อร้าน
    known = {t.lower() for t in brands + couriers + platforms}
    shops = [o for o in ner_orgs if o.lower() not in known]

    # คำนามเฉพาะจาก POS เป็นตัวเลือกสำรองเมื่อ NER ไม่เจออะไรเลย
    if not brands and not shops:
        propn = [w for w, tag in tagged if tag == "PROPN" and len(w) > 2]
        shops = propn[:3]

    dedup = lambda xs: sorted({x.strip() for x in xs if x.strip()})  # noqa: E731
    return {
        "brand": dedup(brands),
        "courier": dedup(couriers),
        "platform": dedup(platforms),
        "seller_shop": dedup(shops),
        "location": dedup(locations),
    }


def guess_product_type(text: str, product_name: str = "") -> str:
    """
    เดาหมวดสินค้าจากคำบ่งชี้ — ดูจากชื่อสินค้าก่อน (ถ้ามี) แล้วค่อยดูจากตัวรีวิว
    ถือเป็น Topic Identification ระดับสินค้า คู่กับ aspect ที่เป็นระดับหัวข้อย่อย
    """
    for source in (product_name or "", text):
        if not source:
            continue
        best, best_pos = "", None
        for category, keywords in cfg.PRODUCT_TYPES.items():
            hits = find_hits(source.lower(), keywords)
            if hits and (best_pos is None or hits[0][1] < best_pos):
                best, best_pos = category, hits[0][1]
        if best:
            return best
    return ""


def extract_all(clean_text: str, normalized: str, tokenized: str, lang: str,
                tagged: list[tuple[str, str]], product_name: str = "") -> dict:
    """รวมผลการสกัดทั้งหมดของกลุ่ม B ใน SCHEMA.md"""
    entities = extract_named_entities(clean_text, tokenized, lang, tagged)
    entities.update({
        "price": extract_price(clean_text),
        "delivery_days": extract_delivery_days(normalized),
        "star_rating": extract_star(clean_text),
        "variant": extract_variant(clean_text),
        "date_mentioned": extract_dates(clean_text, lang),
        "product_type": guess_product_type(tokenized, product_name),
    })
    return entities
