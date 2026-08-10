# -*- coding: utf-8 -*-
"""
ขั้นตอนที่ 4 ของ pipeline: Topic Identification + Aspect-Based Sentiment

จุดต่างจากการวิเคราะห์ความรู้สึกทั่วไป: ระบบนี้ไม่ตอบว่า "รีวิวนี้บวกหรือลบ"
แต่ตอบว่า *ด้านไหน* ดี *ด้านไหน* แย่ เพราะรีวิวจริงส่วนใหญ่เป็นแบบผสม
("ของดีแต่ส่งช้า") ซึ่งการสรุปเป็นขั้วเดียวจะทิ้งข้อมูลที่คนซื้อต้องการที่สุดไป

วิธีที่ใช้: lexicon + หน้าต่างบริบท + การจัดการคำปฏิเสธ
เลือกวิธีนี้แทนโมเดล ML เพราะอธิบายผลได้ทุกขั้นตอน (ชี้ได้ว่าตัดสินจากคำไหน)
และไม่ต้องมีข้อมูลเทรน
"""

from __future__ import annotations

from . import config as cfg
from .matching import drop_substrings, find_hits

Hit = tuple[str, int, int, float]  # (คำ, เริ่ม, จบ, ขั้ว)


def _is_negated(text: str, start: int) -> bool:
    """
    ดูว่าคำนี้ถูกปฏิเสธหรือไม่

    ใช้ระยะเป็น "จำนวนคำที่คั่นกลาง" ไม่ใช่จำนวนตัวอักษร เพราะคำปฏิเสธในภาษาไทย
    เกาะกับคำที่อยู่ถัดไปทันที ถ้าวัดด้วยระยะตัวอักษรอย่างเดียวจะพลิกขั้วผิด เช่น

        "ยัง ไม่ มีปัญหา เลย ทน กว่า ตัวเก่า"

    คำว่า "ไม่" เป็นของ "มีปัญหา" ไม่ใช่ของ "ทน" ที่อยู่ห่างออกไป 2 คำ
    แต่ระยะตัวอักษรระหว่างทั้งคู่มีแค่ ~15 ตัว ซึ่งอยู่ในหน้าต่างเดิม
    """
    window = text[max(0, start - cfg.NEGATION_WINDOW):start]
    for neg in cfg.NEGATIONS:
        index = window.rfind(neg)
        if index == -1:
            continue
        between = window[index + len(neg):]
        if len(between.split()) <= cfg.NEGATION_MAX_GAP:
            return True
    return False


def _build_aspect_lexicon() -> dict[str, list[tuple[str, float]]]:
    """
    รวมพจนานุกรมของทุก aspect เป็นชุดเดียว: คำ -> [(aspect, ขั้ว), ...]

    ต้องรวมเป็นชุดเดียวก่อนค้น ไม่ใช่ค้นทีละ aspect เพราะ find_hits ตัดคำที่ซ้อนกัน
    ได้เฉพาะภายในชุดที่ส่งเข้าไป ถ้าแยกค้นจะเกิดปัญหาสองแบบ:

      ข้ามขั้ว  "ไม่คุ้ม" (ลบ) จะถูกนับซ้อนกับ "คุ้ม" (บวก) แล้วหักล้างกันเป็นศูนย์
      ข้าม aspect "ห่อมาบาง" (บรรจุภัณฑ์ ลบ) มีคำว่า "บาง" (คุณภาพ ลบ) ซ้อนอยู่
                 ทำให้คุณภาพโดนหักคะแนนทั้งที่รีวิวไม่ได้ติคุณภาพ
    """
    lexicon: dict[str, list[tuple[str, float]]] = {}
    for key, spec in cfg.ASPECTS.items():
        for group, polarity in (("pos", 1.0), ("neg", -1.0), ("cues", 0.0)):
            for term in spec[group]:
                entries = lexicon.setdefault(term, [])
                if not any(k == key for k, _ in entries):
                    entries.append((key, polarity))
    return lexicon


ASPECT_LEXICON = _build_aspect_lexicon()


def _lexicon_hits(text: str, positive, negative) -> list[Hit]:
    """ค้นคำแสดงขั้วทั่วไป (ไม่ผูกกับ aspect ใด) พร้อมจัดการคำปฏิเสธ"""
    lexicon: dict[str, float] = {t: 1.0 for t in positive}
    lexicon.update({t: -1.0 for t in negative})  # คำเชิงลบชนะถ้าซ้ำกัน

    hits: list[Hit] = []
    for term, start, end in find_hits(text, lexicon):
        polarity = lexicon[term]
        if polarity != 0.0 and _is_negated(text, start):
            polarity = -polarity
        hits.append((term, start, end, polarity))
    return hits


def _polarity_label(score: float) -> str:
    if score > 0:
        return "pos"
    if score < 0:
        return "neg"
    return "neutral"


def analyze(tokenized_text: str, tagged: list[tuple[str, str]]) -> dict:
    """
    วิเคราะห์ aspect ทั้ง 6 ด้าน

    tokenized_text: ข้อความที่ตัดคำแล้วเชื่อมด้วยช่องว่าง (" ".join(tokens))
                    จำเป็นต้องเป็นรูปแบบนี้ ไม่ใช่ข้อความดิบ เพราะ matching.py
                    ใช้ขอบเขตคำในการกันการแมตช์กลางคำของภาษาไทย

    คืนค่า:
      aspects           - รายชื่อ aspect ที่ถูกพูดถึง
      aspect_sentiment  - {aspect: pos/neg/neutral}
      aspect_evidence   - {aspect: [คำที่ใช้ตัดสิน]} สำหรับแสดงให้ผู้ใช้ตรวจสอบได้
      pros / cons       - คำชม / คำติ ที่สกัดได้
      overall_sentiment - positive / negative / neutral / mixed
      sentiment_score   - -1.0 ถึง 1.0
    """
    text = tokenized_text.lower()

    # ค้นครั้งเดียวด้วยพจนานุกรมรวม เพื่อให้คำที่ยาวกว่าชนะคำที่ซ้อนอยู่ข้างใน
    # ทั้งข้ามขั้วและข้าม aspect
    aspect_hits: dict[str, list[Hit]] = {}
    taken_spans: list[tuple[int, int]] = []

    for term, start, end in find_hits(text, ASPECT_LEXICON):
        negated = _is_negated(text, start)
        taken_spans.append((start, end))
        for key, polarity in ASPECT_LEXICON[term]:
            if polarity != 0.0 and negated:
                polarity = -polarity  # "ไม่พัง" = ดี, "ไม่คุ้ม" = แย่
            aspect_hits.setdefault(key, []).append((term, start, end, polarity))

    # คำแสดงขั้วทั่วไป ("ดี", "แย่", "worst") จะถูกโยงเข้ากับ aspect ที่อยู่ใกล้ที่สุด
    # ข้ามคำที่ทับกับคำเฉพาะของ aspect ไปแล้ว เพื่อไม่ให้นับซ้ำ
    general = _lexicon_hits(text, cfg.GENERAL_POSITIVE, cfg.GENERAL_NEGATIVE)
    general = [h for h in general if not any(s <= h[1] < e for s, e in taken_spans)]

    scores: dict[str, float] = {k: sum(h[3] for h in v) for k, v in aspect_hits.items()}

    for term, start, end, polarity in general:
        if polarity == 0.0:
            continue
        nearest, best_distance = None, cfg.POLARITY_WINDOW
        for key, hits in aspect_hits.items():
            for _, hs, he, _ in hits:
                distance = hs - end if hs > end else start - he
                if 0 <= distance < best_distance:
                    nearest, best_distance = key, distance
        if nearest:
            scores[nearest] += polarity * 0.5

    # ---- สรุปผลราย aspect
    aspect_sentiment = {k: _polarity_label(v) for k, v in scores.items()}
    evidence = {
        k: sorted({h[0] for h in v}, key=len, reverse=True)[:6]
        for k, v in aspect_hits.items()
    }

    # ---- pros / cons จากคำที่ใช้ตัดสินจริง
    pros, cons = [], []
    for hits in aspect_hits.values():
        for term, _, _, polarity in hits:
            (pros if polarity > 0 else cons if polarity < 0 else []).append(term)

    # เสริมด้วยคำคุณศัพท์จาก POS ที่มีขั้วแต่ยังไม่ถูกเก็บ (ใช้ประโยชน์จาก POS tagging)
    for word, tag in tagged:
        low = word.lower()
        if tag != "ADJ" or low in pros or low in cons:
            continue
        if low in cfg.GENERAL_POSITIVE:
            pros.append(word)
        elif low in cfg.GENERAL_NEGATIVE:
            cons.append(word)

    # ลบคำซ้ำ และคำที่เป็นส่วนย่อยของคำอื่น ("บาง" ซ้อนใน "ห่อมาบาง")
    dedup = lambda xs: drop_substrings(list(dict.fromkeys(xs)))  # noqa: E731
    pros, cons = dedup(pros), dedup(cons)

    # ---- สรุปภาพรวม
    n_pos = sum(1 for v in aspect_sentiment.values() if v == "pos")
    n_neg = sum(1 for v in aspect_sentiment.values() if v == "neg")
    general_score = sum(h[3] for h in general)
    total = sum(scores.values()) + general_score * 0.5

    if n_pos and n_neg:
        overall = "mixed"
    elif n_pos or total > 0:
        overall = "positive"
    elif n_neg or total < 0:
        overall = "negative"
    else:
        overall = "neutral"

    magnitude = sum(abs(v) for v in scores.values()) + abs(general_score) * 0.5
    score = round(total / magnitude, 3) if magnitude else 0.0

    return {
        "aspects": list(aspect_hits),
        "aspect_sentiment": aspect_sentiment,
        "aspect_evidence": evidence,
        "pros": pros[:8],
        "cons": cons[:8],
        "overall_sentiment": overall,
        "sentiment_score": score,
    }
