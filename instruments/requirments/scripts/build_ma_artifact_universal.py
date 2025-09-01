#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ma_artifact_universal.py

Генерация сценарного MA-документа (Markdown) из:
- requirements.json (UI-требования из дизайна),
- api_minimal.json (ГОТОВЫЙ минимальный срез API — подаётся на вход),
- style_profile.json (профиль стиля: структура, нумерация, терминология, формат дат, few_shot_examples),
- hints.json (опционально: якорные подсказки по разделам для модели).

Особенности:
- Универсальный план разделов (section_plan): берётся из style_profile.structure.sections,
  иначе выводится из requirements.screens, иначе — дефолт.
- few_shot_examples из style_profile инжектятся в system prompt.
- Блок hints подставляется в user payload из отдельного файла (--hints).
- Никакой логики построения минимального среза из OpenAPI внутри — ожидается готовый api_minimal.json.

Запуск:
  python scripts/build_ma_artifact_universal.py \
    --requirements requirements.json \
    --api-minimal minimal.json \
    --style-profile style/style_profile.json \
    --hints hints.json \
    --out "[MA] Уведомления.md" \
    --model gpt-4o
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Optional LLM
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


# ---------- utils ----------
def sanitize_json_text(text: str) -> str:
    import re
    text = text.lstrip("\ufeff")
    text = re.sub(r'(?m)^\s*//.*$', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r',\s*(\n\s*[}\]])', r'\1', text)
    return text.strip()

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(sanitize_json_text(raw))


# ---------- Section plan (universal) ----------
def derive_feature_name(requirements: Any, api_min: Dict[str, Any]) -> str:
    if isinstance(requirements, dict):
        for k in ("feature", "name", "title"):
            if requirements.get(k):
                return str(requirements[k])
    return str(api_min.get("method_name") or api_min.get("request", {}).get("path") or "Функциональность")

def derive_sections_from_profile(style_profile: Dict[str, Any]) -> Optional[List[str]]:
    return style_profile.get("structure", {}).get("sections") if isinstance(style_profile, dict) else None

def derive_sections_from_requirements(requirements: Any) -> Optional[List[str]]:
    if not isinstance(requirements, dict):
        return None
    screens = requirements.get("screens")
    if not isinstance(screens, list) or not screens:
        return None
    titles = []
    seen = set()
    for s in screens:
        name = (s.get("screen") or s.get("name") or "").strip() if isinstance(s, dict) else ""
        if not name:
            continue
        name = name[:1].upper() + name[1:]
        if name not in seen:
            titles.append(name)
            seen.add(name)
    if not titles:
        return None
    if not any("ошиб" in t.lower() for t in titles):
        titles.append("Обработка ошибок")
    return titles

def default_sections(feature_name: str) -> List[str]:
    return [
        f"Получение данных {feature_name}",
        f"Отображение {feature_name}",
        f"Детальная информация {feature_name}",
        "Обработка ошибок"
    ]

def build_section_plan(requirements: Any, style_profile: Dict[str, Any], api_min: Dict[str, Any]) -> List[str]:
    feature_name = derive_feature_name(requirements, api_min)
    prof = derive_sections_from_profile(style_profile)
    if prof and isinstance(prof, list) and any(prof):
        return [str(s) for s in prof if isinstance(s, str) and s.strip()]
    req = derive_sections_from_requirements(requirements)
    if req:
        return req
    return default_sections(feature_name)


# ---------- Few-shot helpers (inject from style_profile) ----------
def _collect_few_shot_examples_from_profile(style_profile: dict,
                                            max_examples: int = 4,
                                            max_chars: int = 900) -> list[str]:
    """
    Get short examples from style_profile["few_shot_examples"].
    Trim by length, drop empties and duplicates.
    """
    raw = style_profile.get("few_shot_examples") or []
    if not isinstance(raw, list):
        return []
    out, seen = [], set()
    for ex in raw:
        if not isinstance(ex, str):
            continue
        s = ex.strip()
        if not s:
            continue
        s = s[:max_chars].rstrip()
        key = s.lower()
        if key not in seen:
            out.append(s)
            seen.add(key)
        if len(out) >= max_examples:
            break
    return out


def _few_shot_block(style_profile: dict) -> str:
    """
    Create an insertion block for system prompt with few-shot examples.
    Returns empty string if no examples provided.
    """
    examples = _collect_few_shot_examples_from_profile(style_profile)
    if not examples:
        return ""
    parts = ["\n---\nПРИМЕРЫ ОФОРМЛЕНИЯ (few-shot, следовать стилю ниже):\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"Пример {i}:\n{ex}\n")
    parts.append("---\n")
    return "".join(parts)


# ---------- Prompts (UNIVERSAL) ----------
def system_prompt_universal(style_profile: dict) -> str:
    base = """\
Вы — системный аналитик. Сгенерируйте сценарный Markdown-документ требований в стиле профиля.
Строго следуйте списку разделов section_plan — порядок/заголовки менять ЗАПРЕЩЕНО.

Общие правила:
- Пишите поведенчески и нормативно.
- Используйте только данные из api_minimal_json (релевантный срез API).
- Поля API встраивайте в сценарии (условия/действия). Отдельные реестры полей запрещены.
- Если по требованиям нужно поле, которого нет в api_minimal_json — явно пометьте доработку API.
- Соблюдайте нумерацию и формулировки из style_profile_json.
- Если в style_profile_json задан формат дат (например, unix_to_iso) — укажите преобразования.
- Результат: чистый Markdown, без преамбул и пояснений.
"""
    return base + _few_shot_block(style_profile)


def user_prompt_universal(requirements_json: Dict[str, Any],
                          api_minimal_json: Dict[str, Any],
                          style_profile_json: Dict[str, Any],
                          section_plan: List[str],
                          feature_name: str,
                          hints: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        "meta": {
            "feature_name": feature_name,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        },
        "style_profile_json": style_profile_json,
        "requirements_json": requirements_json,
        "api_minimal_json": api_minimal_json,
        "section_plan": section_plan,
        "hints": hints if hints is not None else {},
        "task": (
            "Сформировать единый сценарный MA-документ фронтенда на основе требований и минимального среза API. "
            "Строго следовать section_plan: порядок и заголовки разделов фиксированы. "
            "Описывать поведение экранов/состояний/действий. Поля API использовать только релевантные."
        ),
        "strict_rules": [
            "Не раскрывать Request/Response как реестр полей — только сценарное описание.",
            "Указывать преобразования дат по style_profile_json.",
            "Ясно отмечать несоответствия между требованиями и API, без придумывания данных."
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------- LLM ----------
def call_llm_openai(sys_prompt: str, usr_prompt: str, model: Optional[str], temperature: float) -> str:
    if OpenAI is None:
        raise RuntimeError("Пакет openai не установлен. Установите: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Не задан OPENAI_API_KEY.")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


# ---------- Dry-run ----------
def dry_run_markdown(section_plan: List[str], feature_name: str) -> str:
    out = [f"# Требования\n"]
    for i, sec in enumerate(section_plan, 1):
        out.append(f"## {sec}\n\n")
        out.append(f"1. ({feature_name}) Нормативные требования и сценарии, извлекаемые из входных JSON.\n")
        out.append(f"   1. Подробные шаги и условия.\n")
        out.append(f"      1. Точки интеграции с API и UI.\n\n")
    return "".join(out)


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Универсальная генерация сценарного MA-документа из требований+API+стиля (минимальный срез API подаётся на вход).")
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--api-minimal", required=True, help="Путь к готовому минимальному срезу API (JSON)")
    parser.add_argument("--style-profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hints", help="Путь к hints.json (опционально)")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0.2")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requirements = load_json(args.requirements)
    style_profile = load_json(args.style_profile)
    api_minimal = load_json(args.api_minimal)
    hints = load_json(args.hints) if args.hints else None

    feature_name = derive_feature_name(requirements, api_minimal)
    section_plan = build_section_plan(requirements, style_profile, api_minimal)

    # Optional hardening for notifications naming if no sections provided in style profile
    if "уведомлен" in feature_name.strip().lower() and not style_profile.get("structure", {}).get("sections"):
        section_plan = [
            "Получение уведомлений",
            "Отображение уведомлений",
            "Детальная информация об уведомлении",
            "Обработка ошибок"
        ]

    if args.dry_run:
        md = dry_run_markdown(section_plan, feature_name)
    else:
        sys_prompt = system_prompt_universal(style_profile)
        usr_prompt = user_prompt_universal(requirements, api_minimal, style_profile, section_plan, feature_name, hints)
        try:
            md = call_llm_openai(sys_prompt, usr_prompt, args.model, args.temperature)
        except Exception as e:
            print(f"[LLM ERROR] {e}", file=sys.stderr)
            print("Фолбэк на dry-run.", file=sys.stderr)
            md = dry_run_markdown(section_plan, feature_name)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()
