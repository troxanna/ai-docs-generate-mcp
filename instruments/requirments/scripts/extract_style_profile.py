#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_style_profile.py  (Markdown-only + few_shot_examples)

Назначение:
  Извлечь из набора Markdown-артефактов единый JSON-профиль стиля (style_profile),
  и ДОБАВИТЬ в него поле few_shot_examples — короткие примеры сценарного оформления,
  автоматически извлечённые из самих артефактов (.md/.txt).

Зависимости:
  pip install openai

Переменные окружения:
  OPENAI_API_KEY

Пример запуска:
python scripts/extract_style_profile.py \
  "docs/*.md" \
  -o style/style_profile.json \
  --org "My Company" \
  --lang ru-RU \
  -m gpt-4o-mini \
  --recursive \
  --temperature 0.2 \
  --max-examples 6 \
  --max-example-chars 800
"""
from __future__ import annotations
import argparse, os, sys, json, re
from pathlib import Path
from typing import List, Tuple

# ----------------------- Config ---------------------------
ALLOWED_EXTS = {".md", ".txt"}

# ----------------------- LLM client -----------------------
def _get_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Environment variable OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)

def call_llm(prompt: str, model: str, timeout: int = 120, temperature: float = 0.2) -> str:
    """
    Универсальный вызов с JSON-режимом.
    1) Responses API + json_object
    2) Fallback: Chat Completions + response_format
    3) Fallback: Chat Completions с инструкцией «только JSON»
    """
    client = _get_client()
    # 1) Responses API
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
            temperature=temperature,
            timeout=timeout,
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text
        try:
            return resp.output[0].content[0].text.value  # на случай иных SDK
        except Exception:
            pass
    except TypeError:
        pass
    except Exception:
        pass
    # 2) Chat Completions с JSON mode
    try:
        cmpl = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Верни ТОЛЬКО один валидный JSON-объект без комментариев."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            timeout=timeout,
        )
        return cmpl.choices[0].message.content
    except TypeError:
        # 3) fallback без response_format
        cmpl = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Верни ТОЛЬКО один валидный JSON-объект без комментариев."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            timeout=timeout,
        )
        return (cmpl.choices[0].message.content or "")

# ----------------------- IO -------------------------------
def read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_text(encoding="cp1251", errors="ignore")

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in ALLOWED_EXTS:
        return read_text_file(path)
    raise RuntimeError(f"Unsupported extension for {path.name}. Allowed: {sorted(ALLOWED_EXTS)}")

def collect_paths(inputs: list[str], recursive: bool) -> list[Path]:
    """
    Собирает список путей к файлам по входам: файлы/папки/glob.
    Разрешённые расширения определяются ALLOWED_EXTS.
    """
    out: list[Path] = []
    for raw in inputs:
        q = Path(raw)
        if q.is_file():
            if q.suffix.lower() in ALLOWED_EXTS:
                out.append(q)
        elif q.is_dir():
            it = q.rglob("*") if recursive else q.iterdir()
            for f in it:
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
                    out.append(f)
        else:
            import glob as _glob
            for g in _glob.glob(raw):
                gp = Path(g)
                if gp.is_file() and gp.suffix.lower() in ALLOWED_EXTS:
                    out.append(gp)
    # dedupe + sort
    return sorted({x.resolve() for x in out}, key=lambda z: str(z).lower())

# ----------------------- PII masking ----------------------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{7,}\d)")

def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text

# ----------------------- Prompt (FREE) --------------------
FREE_PROMPT = r"""
РОЛЬ
Ты — аналитик-экстрактор стиля требований. Твоя задача: по корпусу предоставленных Markdown-артефактов сформировать ОДИН объект JSON,
который описывает требования к стилю (tone, формулировки, шаблоны, нормы) так, как это реально наблюдается в корпусе.
Структура JSON не задана заранее — определи набор ключей и вложенность самостоятельно на основе закономерностей корпуса.

ВХОД
ORGANIZATION: {ORG}
LANGUAGE: {LANG}
АРТЕФАКТЫ (Markdown-отрывки, без PII):
<<<НАЧАЛО_КОРПУСА
{ARTIFACTS}
>>>КОНЕЦ_КОРПУСА

ЦЕЛЬ
Сформировать гибкий «профиль стиля требований» в виде JSON:
— какие формулы и модальные глаголы используются;
— как выглядят заголовки/нумерация/идентификаторы;
— какие встречаются шаблоны предложений (напр. «Если …, то …», Gherkin Given/When/Then);
— какие поля и секции обычно присутствуют, как они структурируются;
— типичные единицы/форматы (дат/валют/метрик), обозначения ошибок, лейблы, терминология/глоссарий;
— что запрещено/предпочтительно;
— любые иные устойчивые паттерны, подтверждённые корпусом.

ПРАВИЛА
1) Опираться ТОЛЬКО на предоставленные артефакты: ничего не выдумывай сверх наблюдаемого.
2) Игнорируй персональные и чувствительные данные; не включай их в JSON.
3) Если признак встречается вариативно — отметь это явно (напр., "вариативно": true) или в ключе "notes".
4) Ключи JSON — только на английском в snake_case (lowercase, слова через «_»). Текст значений и примеры — на языке корпуса.
5) По возможности указывай короткие примеры, regex, списки значений, диапазоны длин — ТОЛЬКО если они вытекают из корпуса.
6) Никакого текста вне JSON. Ответ — ровно один валидный JSON-объект.
7) Структуру JSON выбираешь сам — группируй правила так, чтобы ими удобно было пользоваться.
8) Если встречаются API-разделы, не перечисляй эндпоинты; только правила именования и 1–2 примера.
9) Фиксируй только общие правила именования и 1–2 СИНТЕТИЧЕСКИХ примера: entity[].field, entity_id, created_at.
10) НЕ ВКЛЮЧАЙ функциональные/логические требования:
   — поведение UI («при нажатии…», «открываем…», «должно отображаться…»),
   — бизнес-правила и фильтрации данных («исключить атрибуты…», «валидация…»),
   — потоки/интеграции/API («метод get…», «endpoint…», «ответ 4xx/5xx…»),
   — обработку ошибок, авторизацию, сортировки, навигацию.
11) ВКЛЮЧАЙ только нормы оформления текста требований:
    — структура разделов/порядок, заголовки/нумерация, типы списков,
    — форматы дат/единиц/идентификаторов, типографика/капитализация,
    — клише формулировок («Если…, то…», модальные глаголы),
    — лейблы/терминологию и правила их написания.
    Если сомневаешься — НЕ включай.

ВЫХОД
Верни ОДИН валидный JSON-объект со свободной структурой, полностью выведенной из корпуса.
Никаких комментариев, пояснений или Markdown снаружи. Только JSON.
"""

def build_prompt_free(artifacts_text: str, org: str, lang: str) -> str:
    return (FREE_PROMPT
            .replace("{ARTIFACTS}", artifacts_text)
            .replace("{ORG}", org)
            .replace("{LANG}", lang))

# ----------------------- Few-shot extractor (NEW) --------------------
H_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)
NUM_LINE   = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+.+", re.M)     # 1. .... / 1.1. ... / 1.1.1. ...
BULLET_LINE= re.compile(r"^\s*[-–•]\s+.+", re.M)                  # - пункт / – пункт
MODAL_RE   = re.compile(r"\b(должен|должны|выполняется|отображается|необходимо|следует)\b", re.I)

def split_sections(text: str) -> list[tuple[str, str]]:
    """
    Возвращает [(heading, body), ...] по H1..H3. Если заголовки отсутствуют — одна псевдосекция.
    """
    sections: list[tuple[str, str]] = []
    positions = [(m.start(), m.end(), m.group(0), m.group(2)) for m in H_HEADING.finditer(text)]
    if not positions:
        return [("Document", text)]
    for i, (s, e, full, title) in enumerate(positions):
        body_start = e
        body_end = positions[i+1][0] if i+1 < len(positions) else len(text)
        body = text[body_start:body_end]
        sections.append((title.strip(), body))
    return sections

def _compact_block(lines: list[str], max_lines: int = 12) -> str:
    block = "\n".join(lines[:max_lines]).strip()
    return block

def _score_block(block: str) -> int:
    # эвристическая оценка «насколько похоже на сценарный few-shot»
    score = 0
    score += len(NUM_LINE.findall(block)) * 2
    score += len(BULLET_LINE.findall(block))
    if MODAL_RE.search(block):
        score += 3
    # чуть штрафуем слишком длинные блоки
    if len(block) > 1200:
        score -= 1
    return score

def extract_examples_from_text(text: str, max_examples: int = 6, max_chars_per_example: int = 800) -> list[str]:
    examples: list[str] = []
    for title, body in split_sections(text):
        # 1) пробуем нумерованные блоки
        num_matches = list(NUM_LINE.finditer(body))
        if num_matches:
            # собираем локальный фрагмент вокруг первых ~10-15 нумерованных строк
            lines = []
            for m in num_matches[:15]:
                lines.append(m.group(0))
            block = _compact_block(lines, max_lines=14)
            block = block[:max_chars_per_example].rstrip()
            examples.append(block)
        else:
            # 2) пробуем буллеты с модальными глаголами
            bullets = [m.group(0) for m in BULLET_LINE.finditer(body)]
            bullets_modal = [b for b in bullets if MODAL_RE.search(b)]
            if bullets_modal:
                block = _compact_block(bullets_modal, max_lines=10)
                block = block[:max_chars_per_example].rstrip()
                examples.append(block)

    # чистим, маскируем, ранжируем, дедупим
    cleaned: list[str] = []
    seen = set()
    for ex in examples:
        ex = mask_pii(ex)
        ex = ex.strip()
        if not ex or ex.lower() in seen:
            continue
        seen.add(ex.lower())
        cleaned.append(ex)

    # сортировка по «сценарности»
    cleaned.sort(key=_score_block, reverse=True)
    return cleaned[:max_examples]

def post_merge_few_shot(profile: dict, examples: list[str]) -> dict:
    """
    Кладём/сливаем few_shot_examples в профиль.
    """
    if not examples:
        return profile
    existing = profile.get("few_shot_examples")
    merged: list[str] = []
    if isinstance(existing, list):
        merged.extend([str(x).strip() for x in existing if str(x).strip()])
    merged.extend(examples)
    # дедуп
    out: list[str] = []
    seen = set()
    for x in merged:
        key = x.strip().lower()
        if key and key not in seen:
            out.append(x.strip())
            seen.add(key)
    profile["few_shot_examples"] = out
    return profile

# ----------------------- Helpers --------------------------
def _coerce_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def _sanitize_terms(profile: dict, forbidden_terms: list[str]) -> dict:
    """Убираем точные совпадения с именами входных файлов (не ссылаемся на конкретные документы)."""
    def scrub(x):
        if isinstance(x, dict):
            return {k: scrub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        if isinstance(x, str):
            s = x.strip()
            for t in forbidden_terms:
                if s.lower() == t.lower():
                    return ""
            return x
        return x
    return scrub(profile)

# ----------------------- main -----------------------------
def main(argv=None):
    ap = argparse.ArgumentParser("extract_style_profile_md (Markdown only) + few_shot_examples")
    ap.add_argument("inputs", nargs="+", help="Файлы/папки (.md/.txt) или glob-шаблоны")
    ap.add_argument("-o", "--output", default="style_profile.json", help="Куда писать JSON-профиль")
    ap.add_argument("-m", "--model", default="gpt-4o-mini", help="Модель (gpt-4o|gpt-4o-mini|...)")
    ap.add_argument("--org", default="default", help="Название организации/проекта для контекста")
    ap.add_argument("--lang", default="ru-RU", help="Язык артефактов (ru-RU/en-US/...)")
    ap.add_argument("--recursive", action="store_true", help="Рекурсивно сканировать папки")
    ap.add_argument("--max-chars", type=int, default=350_000, help="Предел на общий объём текстов")
    ap.add_argument("--no-mask-pii", action="store_true", help="Отключить простое маскирование PII")
    ap.add_argument("--temperature", type=float, default=0.2, help="Температура модели")
    # NEW:
    ap.add_argument("--max-examples", type=int, default=6, help="Сколько few-shot примеров сохранить")
    ap.add_argument("--max-example-chars", type=int, default=800, help="Обрезать каждый пример до N символов")
    args = ap.parse_args(argv)

    # 1) Сбор входов
    paths = collect_paths(args.inputs, recursive=args.recursive)
    if not paths:
        print("Нет подходящих входных файлов.", file=sys.stderr)
        return 2

    # 2) Чтение
    file_texts: List[Tuple[Path, str]] = []
    for pth in paths:
        try:
            txt = extract_text(pth)
        except Exception as e:
            print(f"[WARN] Пропуск {pth}: {e}", file=sys.stderr)
            continue
        file_texts.append((pth, txt))
    if not file_texts:
        print("Не удалось прочитать документы.", file=sys.stderr)
        return 2

    # 3) Подготовка корпуса
    artifacts_text = ""
    for pth, t in file_texts:
        chunk = t
        if not args.no_mask_pii:
            chunk = mask_pii(chunk)
        artifacts_text += f"===== FILE: {pth.name} =====\n{chunk}\n\n"

    if len(artifacts_text) > args.max_chars:
        artifacts_text = artifacts_text[:args.max_chars] + "\n[TRUNCATED]\n"

    # 3a) Локально извлечём few-shot примеры из всех артефактов (до вызова LLM),
    #     чтобы независимо от ответа модели иметь хорошие примеры.
    few_shots_all: list[str] = []
    for _, raw_text in file_texts:
        if not args.no_mask_pii:
            raw_text = mask_pii(raw_text)
        few_shots_all += extract_examples_from_text(
            raw_text, max_examples=args.max_examples, max_chars_per_example=args.max_example_chars
        )
    # Дедуп по всему корпусу
    _tmp_seen = set()
    unique_few_shots: list[str] = []
    for s in few_shots_all:
        k = s.strip().lower()
        if k and k not in _tmp_seen:
            unique_few_shots.append(s.strip())
            _tmp_seen.add(k)
    # ограничим до max_examples (на случай очень богатого корпуса)
    unique_few_shots = unique_few_shots[:args.max_examples]

    # 4) «Запрещённые» термины (имена файлов без версий/тегов)
    forbidden_terms: list[str] = []
    for pth, _ in file_texts:
        stem = re.sub(r"-v\d+[\w\-]*", "", pth.stem)
        stem = re.sub(r"\[.*?\]\s*", "", stem).strip()
        if stem:
            forbidden_terms.append(stem)

    # 5) Построение промпта
    prompt = build_prompt_free(artifacts_text=artifacts_text, org=args.org, lang=args.lang)

    # 6) Вызов модели
    raw = call_llm(prompt=prompt, model=args.model, temperature=args.temperature)

    # 7) Парсинг JSON
    try:
        data = _coerce_json(raw)
    except Exception:
        Path(args.output + ".raw.txt").write_text(raw, encoding="utf-8")
        raise RuntimeError(f"Model output is not JSON. Saved raw to {args.output}.raw.txt")

    # 8) Пост-обработка
    data = _sanitize_terms(data, forbidden_terms)

    # 8a) Вставим/объединим few_shot_examples
    data = post_merge_few_shot(data, unique_few_shots)

    # 9) Запись
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote style profile → {Path(args.output).resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
