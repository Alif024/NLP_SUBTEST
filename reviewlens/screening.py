# -*- coding: utf-8 -*-
"""
ขั้นตอนที่ 5 ของ pipeline: การคัดกรอง (จุดขายของระบบ)

แนวคิด: ข้อมูลที่ถูก "ลบทิ้ง" ในขั้น cleansing คือสัญญาณที่มีค่าที่สุดในการจับสแปม
เบอร์โทรกับ LINE ID ในรีวิวสินค้าแทบไม่มีเหตุผลอื่นนอกจากการโฆษณาแฝง
โมดูลนี้จึงรับผลจาก cleansing.extract_pii มาใช้ต่อ แทนที่จะทิ้งไป

ผลลัพธ์เป็นคะแนนความน่าเชื่อถือ 0-100 พร้อมเหตุผลที่หักคะแนนทุกข้อ
เพื่อให้ผู้ใช้ตรวจสอบได้ว่าระบบตัดสินจากอะไร
"""

from __future__ import annotations

import hashlib
import re

from . import config as cfg
from .matching import matched_terms

CONTACT_KINDS = ("phone", "line_id", "url", "email", "social")


def text_fingerprint(text: str) -> str:
    """
    ลายนิ้วมือข้อความสำหรับหารีวิวซ้ำ
    ตัดช่องว่างและอักขระพิเศษออกก่อน เพื่อให้จับรีวิวที่ต่างกันแค่วรรคตอนได้ด้วย
    """
    core = re.sub(r"[^\wก-๙]+", "", text.lower())
    return hashlib.sha1(core.encode("utf-8")).hexdigest()


def is_generic(text: str, tokens: list[str]) -> bool:
    """
    รีวิวที่เป็นวลีทั่วไปล้วน ๆ เช่น "ดีมากค่ะ" / "nice product"
    ตัดสินจากจำนวนคำที่เหลือหลังลบวลีทั่วไปออก
    """
    stripped = text.lower()
    for phrase in cfg.GENERIC_PHRASES:
        stripped = stripped.replace(phrase, " ")
    remaining = len(re.findall(r"[a-zA-Zก-๙]{2,}", stripped))
    return len(tokens) <= 6 and remaining <= 1


def screen(
    raw_text: str,
    clean_text: str,
    tokens: list[str],
    pii_removed: dict[str, list[str]],
    aspects: list[str],
    emoji_count: int = 0,
    duplicate_of: str | None = None,
) -> dict:
    """
    ประเมินความน่าเชื่อถือของรีวิว 1 รายการ

    duplicate_of: review_id ของรีวิวที่ข้อความซ้ำกัน (ตรวจในระดับชุดข้อมูล
                  จึงต้องส่งเข้ามาจากภายนอก — ดู pipeline.analyze_batch)
    """
    signals: list[str] = []
    reasons: list[tuple[str, int]] = []
    score = 100

    contacts = {k: v for k, v in pii_removed.items() if k in CONTACT_KINDS}
    promo = matched_terms(clean_text.lower(), cfg.PROMO_KEYWORDS)

    # --- โฆษณาแฝง: มีช่องทางติดต่อ *และ* คำชวนซื้อ
    is_ad_spam = bool(contacts) and bool(promo)
    if is_ad_spam:
        signals.append("contains_contact")
        signals.append("promo_keyword")
        score -= cfg.SCREENING["penalty_ad_spam"]
        reasons.append((f"พบช่องทางติดต่อ ({', '.join(contacts)}) ร่วมกับคำชวนซื้อ ({', '.join(promo[:3])})",
                        cfg.SCREENING["penalty_ad_spam"]))
    elif contacts:
        # มีช่องทางติดต่อแต่ไม่มีคำชวนซื้อ มักเป็นผู้ซื้อทิ้งเบอร์ไว้ให้ร้านติดต่อกลับ
        signals.append("contains_contact")
        score -= cfg.SCREENING["penalty_contact"]
        reasons.append((f"พบข้อมูลติดต่อในรีวิว ({', '.join(contacts)}) — ถูกลบออกจากข้อความแล้ว",
                        cfg.SCREENING["penalty_contact"]))
    elif promo:
        signals.append("promo_keyword")
        score -= cfg.SCREENING["penalty_generic"]
        reasons.append((f"พบคำชวนซื้อ ({', '.join(promo[:3])})", cfg.SCREENING["penalty_generic"]))

    # --- รีวิวสั้นเกินไปจนไม่ให้ข้อมูล
    is_low_quality = len(tokens) < cfg.SCREENING["min_tokens"] and not aspects
    if is_low_quality:
        signals.append("too_short")
        score -= cfg.SCREENING["penalty_low_quality"]
        reasons.append((f"ข้อความสั้นเกินไป ({len(tokens)} คำ) และไม่ระบุด้านใดของสินค้า",
                        cfg.SCREENING["penalty_low_quality"]))

    generic = is_generic(clean_text, tokens)
    if generic and not is_low_quality:
        signals.append("generic_phrase")
        score -= cfg.SCREENING["penalty_generic"]
        reasons.append(("เป็นวลีทั่วไปที่ใช้กับสินค้าอะไรก็ได้", cfg.SCREENING["penalty_generic"]))

    # --- ข้อความซ้ำกับรีวิวอื่นในชุดเดียวกัน
    is_duplicate = duplicate_of is not None
    if is_duplicate:
        signals.append("duplicate")
        score -= cfg.SCREENING["penalty_duplicate"]
        reasons.append((f"ข้อความซ้ำกับรีวิว {duplicate_of}", cfg.SCREENING["penalty_duplicate"]))

    # --- ไม่มี aspect เลยทั้งที่ข้อความยาวพอ
    if not aspects and not is_low_quality:
        signals.append("no_aspect")
        score -= cfg.SCREENING["penalty_no_aspect"]
        reasons.append(("ไม่พบการพูดถึงด้านใดของสินค้าเลย", cfg.SCREENING["penalty_no_aspect"]))

    # --- มีแต่อีโมจิ
    if emoji_count and not re.search(r"[a-zA-Zก-๙]", clean_text):
        signals.append("emoji_only")
        score -= cfg.SCREENING["penalty_low_quality"]
        reasons.append(("รีวิวมีแต่อีโมจิ ไม่มีข้อความ", cfg.SCREENING["penalty_low_quality"]))

    score = max(0, min(100, score))
    if score >= cfg.SCREENING["threshold_pass"]:
        status = cfg.STATUS_PASS
    elif score >= cfg.SCREENING["threshold_suspect"]:
        status = cfg.STATUS_SUSPECT
    else:
        status = cfg.STATUS_SPAM

    return {
        "spam_signals": signals,
        "is_ad_spam": is_ad_spam,
        "is_low_quality": is_low_quality or generic,
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of or "",
        "credibility_score": score,
        "review_status": status,
        "screening_reasons": reasons,
    }
