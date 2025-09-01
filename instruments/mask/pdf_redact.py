#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_redact.py — маскирование ПДн в PDF с помощью DeepPavlov ner_rus_bert + regex.

Дополнительная функциональность:
- Поддержка YAML-файла со стоп-словами/шаблонами, которые НЕЛЬЗЯ маскировать.
  Ключи YAML:
    exact:  # список точных строк (без учёта регистра и лишних пробелов)
      - "ООО Ромашка"
      - "Иван Иванов"
    regex:  # список регулярных выражений (Python/regex), сопоставление без флагов
      - "\\bTest\\d+\\b"
    by_label:  # необязательно: правила для конкретных меток (EMAIL/NAME/PHONE_RU/и т.д.)
      EMAIL:
        regex:
          - "@example\\.com$"
      NAME:
        exact:
          - "Петр Петров"

Примеры:
    python pdf_redact.py in.pdf out.pdf --log log.json \
        --enable-regex --enable-ner \
        --stop-yaml stoplist.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --- regex с поддержкой Юникода, fallback на re ---
try:
    import regex as regx
except Exception:  # pragma: no cover
    import re as regx  # type: ignore

# ---------------------- Утилиты ----------------------

def _norm_str(s: str) -> str:
    """Нормализуем строку для сравнения: трим, схлопывание пробелов, lower."""
    return " ".join((s or "").split()).strip().lower()

# ---------------------- Загрузка стоп-листа ----------------------
class StopList:
    def __init__(self):
        self.global_exact: Set[str] = set()
        self.global_regex: List[regx.Pattern] = []
        # по-меточным правилам: label -> {"exact": set[str], "regex": list[Pattern]}
        self.by_label: Dict[str, Dict[str, object]] = {}

    @staticmethod
    def _compile_many(rx_list: List[str]) -> List[regx.Pattern]:
        out = []
        for pat in rx_list or []:
            try:
                out.append(regx.compile(pat))
            except Exception as e:
                print(f"[WARN] Пропущен некорректный regex из YAML: {pat!r}: {e}", file=sys.stderr)
        return out

    @classmethod
    def from_yaml(cls, path: Path) -> "StopList":
        import yaml  # type: ignore
        sl = cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[ERROR] Не удалось прочитать YAML стоп-слов: {e}", file=sys.stderr)
            sys.exit(4)

        # global exact
        for val in (data.get("exact") or []):
            if isinstance(val, str) and val.strip():
                sl.global_exact.add(_norm_str(val))

        # global regex
        sl.global_regex = cls._compile_many(list(data.get("regex") or []))

        # by_label
        by_label = data.get("by_label") or {}
        if isinstance(by_label, dict):
            for label, section in by_label.items():
                if not isinstance(section, dict):
                    continue
                exact_set: Set[str] = set()
                for val in (section.get("exact") or []):
                    if isinstance(val, str) and val.strip():
                        exact_set.add(_norm_str(val))
                regex_list = cls._compile_many(list(section.get("regex") or []))
                sl.by_label[str(label).upper()] = {"exact": exact_set, "regex": regex_list}
        return sl

    def blocks(self, token: str, label: Optional[str]) -> bool:
        tnorm = _norm_str(token)
        if not tnorm:
            return False
        # глобальные exact/regex
        if tnorm in self.global_exact:
            return True
        for pr in self.global_regex:
            try:
                if pr.search(token):
                    return True
            except Exception:
                continue
        # по-метке
        if label:
            rules = self.by_label.get(label.upper())
            if rules:
                if tnorm in rules.get("exact", set()):
                    return True
                for pr in rules.get("regex", []):
                    try:
                        if pr.search(token):
                            return True
                    except Exception:
                        continue
        return False

# ---------------------- Паттерны ПДн (RU) ----------------------

def build_ru_patterns():
    patterns = [
        ("EMAIL",       r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        ("PHONE_RU",    r"(?:(?:\+7|8)\s?(?:\(\d{3}\)|\d{3})[\s-]?)\d{3}[\s-]?\d{2}[\s-]?\d{2}"),
        ("INN",         r"\b(?:\d{10}|\d{12})\b"),
        ("SNILS",       r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"),
        ("PASSPORT_RF", r"\b\d{2}\s?\d{2}\s?\d{6}\b"),
        ("CARD",        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        ("OGRN",        r"\b\d{13}\b"),
        ("OGRNIP",      r"\b\d{15}\b"),
        ("BIK",         r"\b\d{9}\b"),
        ("IBAN",        r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
        ("SWIFT",       r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
    ]
    return [(name, regx.compile(rx)) for name, rx in patterns]

# ---------------------- ФИО-шаблоны ----------------------

def fio_matchers():
    cyr_particle = r"(?:(?:[дД]е|[фФ]он)\s+)?"
    lat_particle = r"(?:(?i:(?:o'|d'|de|van|von|da|di|du|del|della|de\s+la)\s+))?"
    cyr_name = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
    lat_name = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"

    fio3_cyr = regx.compile(r"\b" + cyr_particle + cyr_name + r"\s+" + cyr_name + r"\s+" + cyr_name + r"\b")
    fio2_cyr = regx.compile(r"\b" + cyr_particle + cyr_name + r"\s+" + cyr_name + r"\b")
    init2_cyr = regx.compile(r"\b" + cyr_particle + cyr_name + r"\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.\b")
    init1_cyr = regx.compile(r"\b" + cyr_particle + cyr_name + r"\s+[А-ЯЁ]\.\b")

    fio3_lat = regx.compile(r"\b" + lat_particle + lat_name + r"\s+" + lat_name + r"\s+" + lat_name + r"\b", regx.I)
    fio2_lat = regx.compile(r"\b" + lat_particle + lat_name + r"\s+" + lat_name + r"\b", regx.I)
    init2_lat = regx.compile(r"\b" + lat_particle + lat_name + r"\s+[A-Z]\.\s*[A-Z]\.\b", regx.I)
    init1_lat = regx.compile(r"\b" + lat_particle + lat_name + r"\s+[A-Z]\.\b", regx.I)

    return dict(
        fio3_cyr=fio3_cyr, fio2_cyr=fio2_cyr, init2_cyr=init2_cyr, init1_cyr=init1_cyr,
        fio3_lat=fio3_lat, fio2_lat=fio2_lat, init2_lat=init2_lat, init1_lat=init1_lat
    )

# ---------------------- Сбор PERSON из BIO-тегов ----------------------

def collect_person_spans(tokens, tags):
    """
    Собирает строки PERSON из BIO-тегов (B-PER / I-PER / O).
    Возвращает список строк (без доп. нормализации).
    """
    spans = []
    cur = []
    for tok, tag in zip(tokens, tags):
        tag = tag.upper()
        if tag.endswith("PER"):
            if tag.startswith("B-") or (cur and tag.startswith("B-")):
                if cur:
                    spans.append(" ".join(cur))
                    cur = []
                cur = [tok]
            elif tag.startswith("I-"):
                if not cur:
                    cur = [tok]
                else:
                    cur.append(tok)
            else:
                # просто "PER" без префикса — на всякий
                cur.append(tok)
        else:
            if cur:
                spans.append(" ".join(cur))
                cur = []
    if cur:
        spans.append(" ".join(cur))
    # мини-чистка двойных пробелов
    return [" ".join(s.split()) for s in spans if s.strip()]

# ---------------------- NER через DeepPavlov ----------------------

def build_ner():
    from deeppavlov import configs, build_model
    # загрузит/скачает модель по конфигу ner_rus_bert
    ner_model = build_model(configs.ner.ner_rus_bert, download=True)
    return ner_model

# ---------------------- Основной скрипт ----------------------

def main():
    ap = argparse.ArgumentParser(description="Redact PII in PDF using DeepPavlov ner_rus_bert + regex (RU).")
    ap.add_argument("input", help="Входной PDF")
    ap.add_argument("output", help="Выходной (редактированный) PDF")
    ap.add_argument("--log", required=True, help="Путь к JSON-логу")
    ap.add_argument("--enable-ner", action="store_true", help="Включить DeepPavlov NER (PERSON)")
    ap.add_argument("--enable-regex", action="store_true", default=True, help="Включить regex-паттерны ПДн (по умолчанию вкл.)")
    # управление редактированием содержимого
    ap.add_argument("--images-mode", type=int, default=2, help="images режим в apply_redactions (0..3), по умолчанию 2")
    ap.add_argument("--graphics-mode", type=int, default=2, help="graphics режим (0..2), по умолчанию 2")
    ap.add_argument("--text-mode", type=int, default=0, help="text режим (0..1), по умолчанию 0 (удалять)")
    ap.add_argument("--min-font", type=float, default=5.0, help="минимальная высота bbox для учёта совпадения")
    ap.add_argument("--show-token", action="store_true", help="печать заменителя поверх (иначе сплошная заливка)")
    ap.add_argument("--replacement", default="", help="текст заменителя при --show-token")
    ap.add_argument("--fio-allow-two", action="store_true", default=True, help="Фамилия Имя")
    ap.add_argument("--fio-allow-initials", action="store_true", default=True, help="Фамилия И.О./Фамилия И.")
    ap.add_argument("--fio-allow-latin", action="store_true", default=True, help="латиница (Ivanov Ivan / Ivanov I.)")
    # Новое: стоп-лист из YAML
    ap.add_argument("--stop-yaml", dest="stop_yaml", help="Путь к YAML со стоп-словами/шаблонами (запрещено маскировать)")

    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    log_path = Path(args.log)

    if not in_path.exists():
        print(f"[ERROR] Файл не найден: {in_path}", file=sys.stderr)
        sys.exit(2)

    # Импорт PyMuPDF
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        print("[ERROR] Не удалось импортировать PyMuPDF (пакет 'pymupdf').", file=sys.stderr)
        sys.exit(3)

    # Подготовка
    matcher = fio_matchers()
    compiled_patterns = build_ru_patterns() if args.enable_regex else []
    ner = None
    if args.enable_ner:
        try:
            ner = build_ner()
        except Exception as e:
            print(f"[WARN] Не удалось инициализировать DeepPavlov NER: {e}", file=sys.stderr)

    stoplist = None
    if args.stop_yaml:
        try:
            stoplist = StopList.from_yaml(Path(args.stop_yaml))
        except ModuleNotFoundError:
            print("[ERROR] Для параметра --stop-yaml требуется пакет PyYAML (pip install pyyaml)", file=sys.stderr)
            sys.exit(5)

    all_logs = []

    with fitz.open(str(in_path)) as doc:
        for page_index, page in enumerate(doc):
            page_text = page.get_text("text") or ""
            found_items: List[Tuple[str, str, str]] = []  # (text, label, source)

            # ---- REGEX ПДн ----
            if compiled_patterns and page_text.strip():
                for name, patt in compiled_patterns:
                    for m in patt.finditer(page_text):
                        val = m.group(0).strip()
                        if val:
                            found_items.append((val, name, "regex"))

            # ---- NER (PERSON) через DeepPavlov ----
            if ner and page_text.strip():
                # В DeepPavlov удобнее подавать текст построчно
                lines = [ln for ln in page_text.splitlines() if ln.strip()]
                for ln in lines:
                    try:
                        tokens, tags = ner([ln])
                        if not tokens or not tags:
                            continue
                        tokens = tokens[0]
                        tags = tags[0]
                        spans = collect_person_spans(tokens, tags)  # строки PERSON
                        for s in spans:
                            exacts = []
                            for mm in matcher["fio3_cyr"].finditer(s): exacts.append(mm.group(0))
                            for mm in matcher["fio3_lat"].finditer(s): exacts.append(mm.group(0))
                            if args.fio_allow_two:
                                for mm in matcher["fio2_cyr"].finditer(s): exacts.append(mm.group(0))
                                for mm in matcher["fio2_lat"].finditer(s): exacts.append(mm.group(0))
                            if args.fio_allow_initials:
                                for mm in matcher["init2_cyr"].finditer(s): exacts.append(mm.group(0))
                                for mm in matcher["init1_cyr"].finditer(s): exacts.append(mm.group(0))
                                for mm in matcher["init2_lat"].finditer(s): exacts.append(mm.group(0))
                                for mm in matcher["init1_lat"].finditer(s): exacts.append(mm.group(0))
                            if exacts:
                                for ex in exacts:
                                    found_items.append((ex, "NAME", "ner"))
                            else:
                                found_items.append((" ".join(s.split()), "NAME", "ner"))
                    except Exception:
                        continue

            # ---- Применяем стоп-лист: фильтруем найденные элементы ----
            if stoplist:
                filtered_items: List[Tuple[str, str, str]] = []
                skipped = 0
                for token, label, source in found_items:
                    if stoplist.blocks(token, label):
                        skipped += 1
                        continue
                    filtered_items.append((token, label, source))
                found_items = filtered_items
            else:
                skipped = 0

            # ---- Превращаем найденные строки в прямоугольники и редактируем ----
            page_log = []
            seen_rects = set()

            for token, label, source in found_items:
                # 1) прямой поиск
                rects = page.search_for(token)
                # 2) «мягкий» поиск
                if not rects:
                    rects = page.search_for(token, hit_max=100, quads=False)
                # 3) поиск по нормализованным пробелам
                if not rects:
                    token_norm = " ".join(token.split())
                    if token_norm and token_norm != token:
                        rects = page.search_for(token_norm) or page.search_for(token_norm, hit_max=100, quads=False)

                for r in rects or []:
                    if (r.y1 - r.y0) < args.min_font:
                        continue
                    sig = (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2), token)
                    if sig in seen_rects:
                        continue
                    seen_rects.add(sig)

                    if args.show_token:
                        page.add_redact_annot(r, text=(args.replacement or token), fill=(0, 0, 0))
                    else:
                        page.add_redact_annot(r, fill=(0, 0, 0))

                    page_log.append({
                        "page": page_index + 1,
                        "label": label,
                        "source": source,
                        "text": token,
                        "rect": [r.x0, r.y0, r.x1, r.y1],
                    })

            if page_log:
                page.apply_redactions(images=args.images_mode, graphics=args.graphics_mode, text=args.text_mode)

            # добавим информацию о пропусках по стоп-листу
            if skipped:
                page_log.append({
                    "page": page_index + 1,
                    "skipped_by_stoplist": int(skipped)
                })

            all_logs.extend(page_log)

        doc.save(str(out_path), garbage=4, deflate=True, clean=True, incremental=False)

    # ---- ЛОГ ----
    payload = {
        "input": str(in_path),
        "output": str(out_path),
        "ner": bool(ner is not None),
        "regex_enabled": bool(compiled_patterns),
        "stop_yaml": str(args.stop_yaml or ""),
        "redactions": all_logs,
    }
    Path(log_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✔ Готово: {out_path}\n📝 Лог: {log_path}\nРедакций: {len([x for x in all_logs if 'rect' in x])}")


if __name__ == "__main__":
    main()
