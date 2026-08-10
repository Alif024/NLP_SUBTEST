# -*- coding: utf-8 -*-
"""
ตัวช่วยค้นคำในข้อความ ใช้ร่วมกันระหว่าง extract.py และ aspects.py

ปัญหาที่ต้องแก้: ภาษาไทยไม่เว้นวรรคระหว่างคำ ทำให้ `\\b` ของ regex ใช้ไม่ได้
ถ้าค้นแบบ substring ตรง ๆ จะเจอผลผิดเพียบ เช่น
    "ทน" (ทนทาน)  ไปแมตช์กลางคำ "ตัวแทน"
    "เลย" (จังหวัด) ไปแมตช์ใน "ได้เลยค่ะ"

วิธีแก้: ค้นบนข้อความที่ตัดคำแล้วและเชื่อมด้วยช่องว่าง ("รับ ตัวแทน จำหน่าย")
โดยบังคับว่าคำค้นต้อง **เริ่ม** ที่ขอบเขตคำ แต่ยอมให้จบกลางคำได้
  - บังคับจุดเริ่ม -> ตัดผลผิดแบบ "ตัวแทน" ทิ้ง เพราะ "ทน" อยู่กลางโทเคน
  - ไม่บังคับจุดจบ -> ยังจับ "ดี" ใน "ดีมาก" ได้ ซึ่งเป็นคำเดียวที่ newmm รวมไว้
และยอมให้มีช่องว่างคั่นกลางคำค้นได้ เพราะ "ส่งช้า" ถูกตัดเป็น "ส่ง ช้า"
"""

from __future__ import annotations

import re
from functools import lru_cache

from .config import THAI_CHAR


@lru_cache(maxsize=8192)
def _compile(term: str) -> re.Pattern:
    if not THAI_CHAR.search(term):
        return re.compile(rf"\b{re.escape(term)}\b", re.I)

    # ยอมให้มีช่องว่างแทรกระหว่างอักขระ เพราะข้อความถูกตัดคำมาแล้ว
    body = r"\s*".join(re.escape(ch) for ch in term)
    return re.compile(rf"(?:(?<=\s)|^){body}")


def find_hits(text: str, terms) -> list[tuple[str, int, int]]:
    """
    คืนรายการ (คำที่เจอ, ตำแหน่งเริ่ม, ตำแหน่งจบ) ของทุกคำใน terms ที่พบในข้อความ

    ถ้าคำหนึ่งซ้อนอยู่ในอีกคำที่ยาวกว่า จะเก็บเฉพาะคำที่ยาวกว่า
    เพื่อไม่ให้ "คุ้ม" ถูกนับเป็นบวกซ้อนกับ "ไม่คุ้ม" ที่เป็นลบ
    """
    hits: list[tuple[str, int, int]] = []
    for term in terms:
        for m in _compile(term).finditer(text):
            hits.append((term, m.start(), m.end()))

    hits.sort(key=lambda h: (h[1], -(h[2] - h[1])))
    kept: list[tuple[str, int, int]] = []
    for hit in hits:
        if any(k[1] <= hit[1] and hit[2] <= k[2] for k in kept):
            continue  # ถูกครอบด้วยคำที่ยาวกว่าแล้ว
        kept.append(hit)
    return kept


def contains(text: str, terms) -> bool:
    return any(_compile(t).search(text) for t in terms)


def matched_terms(text: str, terms) -> list[str]:
    return [t for t, _, _ in find_hits(text, terms)]


def drop_substrings(terms: list[str]) -> list[str]:
    """
    ลบคำที่เป็นส่วนย่อยของคำอื่นในรายการเดียวกัน

    ใช้กับ pros/cons ที่รวมผลจากหลาย aspect เข้าด้วยกัน
    เช่น "บาง" (คุณภาพ) กับ "ห่อมาบาง" (บรรจุภัณฑ์) ควรเหลือแค่คำที่ยาวกว่า
    """
    ordered = sorted(terms, key=len, reverse=True)
    kept: list[str] = []
    for term in ordered:
        if not any(term in longer for longer in kept):
            kept.append(term)
    return [t for t in terms if t in kept]
