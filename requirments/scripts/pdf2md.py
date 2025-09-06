#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Как запустить:
 python scripts/pdf2md.py \
  --example log/example.md \
  --section "Требования" \
  --model gpt-4o \
  docs/raw \
  --out-dir .
"""

import argparse
import os
import re
from pathlib import Path
from typing import List
from pypdf import PdfReader

# --- LLM client (OpenAI Python SDK) ---
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Не найден пакет 'openai'. Установите: pip install openai")

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")  # можно переопределить переменной окружения


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(chunks).strip()


def strip_links(md: str) -> str:
    md = re.sub(r"\[([^\]]+)\]\((?!#)[^)]+\)", r"\1", md)      # [текст](url)
    md = re.sub(r"<https?://[^>]+>", "", md)                   # <http://...>
    md = re.sub(r"(?<!\()https?://\S+", "", md)                # голые URL
    return md


def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r'[\\/:*?"<>|]+', "-", name)
    return name or "section"


def build_system_prompt(example_md_text: str, section_title: str) -> str:
    return (
        "Твоя задача — ПРЕОБРАЗОВАНИЕ ТЕКСТА ДОКУМЕНТА В MARKDOWN по строгим правилам:\n"
        f"1) Возьми ТОЛЬКО раздел с названием «{section_title}» из переданного содержимого PDF.\n"
        "   • Заголовок может отличаться регистром и иметь префикс с номером (например, «1. Требования»).\n"
        "   • Границы раздела: от этого заголовка до следующего заголовка того же или более высокого уровня.\n"
        "2) Преобразуй найденный раздел в ОТДЕЛЬНЫЙ Markdown.\n"
        "3) Полностью УДАЛИ ВСЕ ССЫЛКИ.\n"
        "4) Корректно обработай таблицы и нумерованные списки.\n"
        "5) Если список находится ВНУТРИ ТАБЛИЦЫ — замени таблицу на Заголовок и Описание (внутри — нумерованный список).\n"
        "6) Верни ТОЛЬКО целевой Markdown, без преамбулы и комментариев.\n\n"
        "===== НАЧАЛО ПРИМЕРА example.md =====\n"
        f"{example_md_text}\n"
        "===== КОНЕЦ ПРИМЕРА example.md =====\n"
    )


def call_llm(model: str, system_prompt: str, pdf_text: str, section_title: str) -> str:
    client = OpenAI()
    user_prompt = (
        "Ниже передано «сырьё» — плоский текст, извлечённый из PDF. "
        f"Сначала найди раздел с заголовком «{section_title}», затем выполни преобразование по правилам. "
        "Верни только Markdown.\n\n"
        "===== НАЧАЛО ИСТОЧНИКА =====\n"
        f"{pdf_text}\n"
        "===== КОНЕЦ ИСТОЧНИКА ====="
    )

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        )
        try:
            text = resp.output_text
        except AttributeError:
            text = resp.output[0].content[0].text
        return text.strip()
    except Exception:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (chat.choices[0].message.content or "").strip()


def collect_pdfs(sources: List[Path], recursive: bool) -> List[Path]:
    """Принимает файлы и/или папки. Возвращает список PDF."""
    pdfs: List[Path] = []
    for src in sources:
        p = Path(src).expanduser()
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        elif p.is_dir():
            it = p.rglob("*.pdf") if recursive else p.glob("*.pdf")
            pdfs.extend(sorted(it))
        else:
            # Если оболочка не раскрыла шаблон, а путь не существует — пропускаем
            continue
    # убираем дубликаты
    seen = set()
    unique = []
    for f in pdfs:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(f)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Извлечь из PDF указанный раздел (например, «Требования») с помощью LLM и сохранить в Markdown."
    )
    parser.add_argument("--example", required=True, type=Path, help="Путь к example.md (эталон структуры).")
    parser.add_argument("--section", required=True, help="Название блока/раздела для извлечения (например: Требования).")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Имя модели LLM (по умолчанию: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out_md"),
        help="Каталог для результата (по умолчанию: ./out_md).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Рекурсивный обход вложенных папок при передаче каталогов.",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Один или несколько путей: PDF-файлы и/или папки с PDF.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Не задан OPENAI_API_KEY")

    example_md_text = args.example.read_text(encoding="utf-8")
    system_prompt = build_system_prompt(example_md_text, args.section)

    pdf_list = collect_pdfs(args.sources, args.recursive)
    if not pdf_list:
        raise SystemExit("PDF не найдены. Проверьте пути/маски или используйте --recursive.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    max_chars = int(os.environ.get("PDF_MAX_CHARS", "400000"))
    for pdf_path in pdf_list:
        print(f"[i] Обработка: {pdf_path}")
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print(f"[!] Не удалось извлечь текст: {pdf_path.name}")
            continue
        if len(raw_text) > max_chars:
            raw_text = raw_text[:max_chars]

        md = call_llm(args.model, system_prompt, raw_text, args.section)
        md = strip_links(md).strip()

        base = pdf_path.with_suffix("").name
        sect = safe_filename(args.section)
        out_path = args.out_dir / f"{base}.{sect}.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"[✓] Сохранено: {out_path}")

    print("[✓] Готово.")


if __name__ == "__main__":
    main()
