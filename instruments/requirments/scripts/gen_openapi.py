#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_openapi_from_requirements.py (v5)
Теперь поддерживаются ОБА варианта источника UI:
  1) --ui-json  — JSON из screens_to_requirements.py (структурированные требования по экранам)
  2) --design / --design-dir [--design-recursive] — сами скриншоты дизайна (изображения)

Скрипт соберёт всё вместе с бизнес‑требованиями и контекстом и попросит LLM сгенерировать OpenAPI.
Если переданы и изображения, и ui‑json — они оба добавляются в промпт (в виде текста и картинок).

Зависимости:
  pip install openai pyyaml
  export OPENAI_API_KEY=...

Пример (только скриншоты):
  python gen_openapi_from_requirements.py \
    --requirements source/api_requirements.md \
    --design-dir screens \
    --model gpt-4o \
    --out api.yaml

Пример (ui‑json + скриншоты + контекст):
  python gen_openapi_from_requirements.py \
    --requirements source/api_requirements.md \
    --ui-json source/requirements.json \
    --design-dir screens --design-recursive \
    --context source/api_ctx.json \
    --model gpt-4o \
    --out api.yaml
"""
from __future__ import annotations
import argparse, json, os, sys, base64, mimetypes
from pathlib import Path
from typing import Any, Dict, List, Iterable

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

SYSTEM_PROMPT = """Ты — эксперт по проектированию и описанию API.  
Твоя задача — помогать в создании и проверке OpenAPI-спецификаций.  
Учитывайте:
- Требования к проектированию API, UI‑требования (JSON) и/или сами изображения экранов (скриншоты).
- Для изображений: извлеките ключевые сущности и действия UI (списки, поля, фильтры, сортировка, навигация) и спроецируйте их на API.
- Если контекст требует JWT — добавьте bearer securityScheme (JWT) и примените его.
- Указывайте: paths, операции, параметры (path/query), requestBody (если нужен), responses с JSON‑схемами, components.schemas, security.
- Добавляй enum для перечислений (например, для поля currency можно создать enum с возможными значениями RUB, USD, EUR). Значения enum должны быть на английском языке.
Ответ — только OpenAPI (YAML или JSON), без комментариев и пояснений вокруг. Описание (description) полей и методов в спецификации должны быть на русском языке.
"""

USER_INSTRUCTIONS = """Сформируй OpenAPI 3.0+ (YAML/JSON) из входных материалов.
Если часть спецификации неочевидна — выбери минимально достаточный и однозначный вариант.
Если контекст указывает количество методов — соблюдай его.
Ответ — только OpenAPI (YAML/JSON)."""

# ---------- OpenAI ----------
def _init_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        print("❌ Требуется пакет 'openai' (pip install openai)", file=sys.stderr)
        raise
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не найден в окружении")
    return OpenAI(api_key=api_key)

# ---------- IO helpers ----------
def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _load_requirements(path: Path) -> str:
    try:
        return _read_text_file(path)
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать требования из {path}: {e}")

def _load_json(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Файл не найден: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Не удалось распарсить JSON {path}: {e}")
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _iter_images_in_dir(folder: Path, recursive: bool = False) -> Iterable[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    for p in folder.glob(pattern):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p.resolve()

def _collect_design_paths(design_args: List[str] | None,
                          design_dir: str | None,
                          recursive: bool) -> List[Path]:
    paths: List[Path] = []
    if design_dir:
        dir_path = Path(design_dir).resolve()
        if not dir_path.exists():
            print(f"⚠️  Папка дизайнов не найдена: {dir_path}", file=sys.stderr)
        else:
            paths.extend(sorted(_iter_images_in_dir(dir_path, recursive=recursive)))
    for d in (design_args or []):
        p = Path(d).resolve()
        if not p.exists():
            print(f"⚠️  Файл дизайна не найден и будет пропущен: {p}", file=sys.stderr)
            continue
        paths.append(p)
    # dedupe
    uniq, seen = [], set()
    for p in paths:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def _read_image_as_data_url(p: Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _extract_code_block(text: str) -> str:
    import re
    blocks = re.findall(r"```(?:yaml|yml|json)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0].strip()
    return text.strip()

def _normalize_openapi(text: str) -> str:
    s = text.strip()
    # JSON?
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # YAML?
    if yaml is not None:
        try:
            obj = yaml.safe_load(s)
            return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
        except Exception:
            pass
    return s

# ---------- Messages ----------
def build_messages(req_text: str,
                   ui_json: str | None,
                   ctx_json: str | None,
                   image_paths: List[Path]) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": USER_INSTRUCTIONS},
                                          {"type": "text", "text": "ТРЕБОВАНИЯ К ПРОЕКТИРОВАНИЮ API:\n" + req_text}]
    if ui_json:
        user_content.append({"type": "text", "text": "UI‑ТРЕБОВАНИЯ (JSON):\n" + ui_json})
    if ctx_json:
        user_content.append({"type": "text", "text": "ДОП. КОНТЕКСТ (JSON):\n" + ctx_json})
    if image_paths:
        user_content.append({"type": "text", "text": "ДИЗАЙН (изображения): ниже прикреплены скриншоты."})
        for p in image_paths:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": _read_image_as_data_url(p), "detail": "high"}
            })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

# ---------- Main ----------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("Generate OpenAPI via LLM from requirements + (ui-json and/or screenshots) + context")
    ap.add_argument("--requirements", required=True, help="Файл с бизнес‑требованиями (md/txt/json/yaml)")
    ap.add_argument("--ui-json", help="Файл UI‑требований (JSON из screens_to_requirements.py)")
    ap.add_argument("--design", action="append", default=[], help="Путь к скриншоту дизайна (можно несколько раз)")
    ap.add_argument("--design-dir", help="Папка со скриншотами дизайна")
    ap.add_argument("--design-recursive", action="store_true", help="Рекурсивный поиск изображений в подпапках")
    ap.add_argument("--context", help="JSON‑файл с доп. контекстом (платформа, JWT, число методов и т.п.)")
    ap.add_argument("--model", required=True, help="Имя модели (напр., gpt-4o, gpt-4o-mini)")
    ap.add_argument("--out", required=True, help="Куда сохранить OpenAPI (yaml/json)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true", help="Не вызывать LLM, только сохранить подготовленный ввод")
    args = ap.parse_args(argv)

    req_path = Path(args.requirements).resolve()
    if not req_path.exists():
        print(f"❌ Не найден файл требований: {req_path}", file=sys.stderr)
        return 2

    # inputs
    req_text = _load_requirements(req_path)
    ui_json = None
    if args.ui_json:
        try:
            ui_json = _load_json(Path(args.ui_json).resolve())
        except Exception as e:
            print(f"⚠️  UI‑JSON не загружен: {e}", file=sys.stderr)
    ctx_json = None
    if args.context:
        try:
            ctx_json = _load_json(Path(args.context).resolve())
        except Exception as e:
            print(f"⚠️  CONTEXT‑JSON не загружен: {e}", file=sys.stderr)

    images = _collect_design_paths(args.design, args.design_dir, args.design_recursive)

    if not images and not ui_json:
        print("ℹ️  Не переданы ни --ui-json, ни изображения дизайна. Модель будет опираться только на бизнес‑требования.", file=sys.stderr)

    messages = build_messages(req_text, ui_json, ctx_json, images)

    if args.dry_run:
        debug = {
            "system": SYSTEM_PROMPT,
            "user_messages": messages[1]["content"],
            "model": args.model,
            "temperature": args.temperature,
            "images_count": len(images),
        }
        Path(args.out).write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📝 DRY-RUN: подготовленные сообщения сохранены в {args.out}")
        return 0

    # LLM call
    try:
        client = _init_openai()
        resp = client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            messages=messages,
        )
        content = (resp.choices[0].message.content or "").strip()
        code = _extract_code_block(content)
        normalized = _normalize_openapi(code)
        Path(args.out).write_text(normalized, encoding="utf-8")
        print(f"✅ OpenAPI сохранён в {args.out}")
        return 0
    except Exception as e:
        fallback = str(Path(args.out).with_suffix(".error.json"))
        debug = {
            "error": str(e),
            "system": SYSTEM_PROMPT,
            "user_messages": messages[1]["content"],
            "model": args.model,
            "temperature": args.temperature,
        }
        Path(fallback).write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"❌ Ошибка LLM: {e}\nℹ️ Отладочный дамп сохранён в {fallback}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
