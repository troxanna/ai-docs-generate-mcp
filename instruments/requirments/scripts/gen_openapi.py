#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_openapi.py — добавление НОВЫХ методов в существующий OpenAPI (финальная версия)

Что делает:
- Читает исходный OpenAPI (YAML/JSON) как «образец стиля».
- Принимает ОБЯЗАТЕЛЬНО: --context (JSON) и дизайн (через --design/--design-dir; нужно хотя бы одно).
- Опционально принимает: --requirements (жёсткие ограничения/правила для LLM).
- Просит LLM сгенерировать ТОЛЬКО фрагмент с новыми methods/paths и связанными components.
- Аккуратно мержит фрагмент в существующую спецификацию (аддитивно; политика конфликтов настраивается).
- Сохраняет формат исходника (если вход был YAML — на выходе YAML; если JSON — JSON).
- Политика по operationId: по умолчанию AUTOGEN — для новых операций авто-сгенерирует operationId в стиле, близком к исходнику.
- Новая опция: --gen-stoplight — автогенерация x-stoplight.id для новых узлов (paths/operations/components).

Зависимости:
  pip install openai pyyaml

Примеры запуска (минимум, без requirements)
zsh/bash:
    python scripts/gen_openapi.py \
    --in-openapi spec/спека.yaml \
    --out ./api.yaml \
    --model gpt-4o-mini \
    --context ./source/api_ctx.json \
    --design-dir ./screens \
    --gen-stoplight auto \
    --conflict-policy skip \
    --in-place
"""
from __future__ import annotations
import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

# ===================== PROMPTS =====================
SYSTEM_PROMPT = """Ты — эксперт по проектированию API и оформлению OpenAPI.
Тебе дают текущую спецификацию OpenAPI (это ОБРАЗЕЦ СТИЛЯ), набор скриншотов дизайна и контекст (JSON).
Твоя задача — предложить ТОЛЬКО ДОБАВЛЕНИЯ: новые paths и связанные components
(schemas/parameters/responses/securitySchemes/requestBodies/headers), необходимые для реализации новых возможностей.

Правила:
- Возвращай ТОЛЬКО фрагмент OpenAPI, пригодный для аддитивного мерджа без удаления/изменения существующих узлов.
- Поддержи стиль входной спеки: формат (YAML/JSON), язык описаний, соглашения по именам и структуру.
- Строго избегай вольного текста вне структуры OpenAPI. Ответ — только код фрагмента.
"""

USER_INSTRUCTIONS = """Сформируй ДОБАВЛЕНИЯ к OpenAPI (только новые методы и связанные компоненты).

Правила приоритета:
1. ОБРАЗЕЦ СПЕЦИФИКАЦИИ — главный источник стилистики и формата.
2. CONTEXT (JSON) — база знаний, словарь терминов, справочные описания.
3. REQUIREMENTS (если переданы) — это ⚠️ ЖЁСТКИЕ ОГРАНИЧЕНИЯ. Их нужно строго соблюдать даже в ущерб удобству.
4. Не удаляй и не изменяй существующие узлы, только добавляй новые.

Ответ — только OpenAPI-фрагмент (YAML/JSON), готовый к мерджу.
"""

# ===================== HELPERS =====================
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_text_or_fail(path: Path, what: str) -> str:
    try:
        return _read_text(path)
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать {what} из {path}: {e}")


def _load_json_and_dump(path: Path) -> str:
    obj = json.loads(_read_text(path))
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _iter_images(folder: Path, recursive: bool) -> Iterable[Path]:
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
        d = Path(design_dir).resolve()
        if d.exists():
            paths.extend(sorted(_iter_images(d, recursive)))
        else:
            print(f"⚠️  Папка дизайнов не найдена: {d}", file=sys.stderr)
    for it in (design_args or []):
        p = Path(it).resolve()
        if p.exists():
            paths.append(p)
        else:
            print(f"⚠️  Файл дизайна не найден и будет пропущен: {p}", file=sys.stderr)
    # dedupe
    uniq, seen = [], set()
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _img_as_data_url(p: Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_code_block(text: str) -> str:
    blocks = re.findall(r"```(?:yaml|yml|json)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (blocks[0] if blocks else text).strip()


def _norm_to_format(text: str, prefer_yaml: bool) -> str:
    s = text.strip()
    # JSON?
    try:
        obj = json.loads(s)
        if prefer_yaml and yaml is not None:
            return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # YAML?
    if yaml is not None:
        try:
            obj = yaml.safe_load(s)
            if prefer_yaml:
                return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return s


def _parse_openapi(text: str) -> Tuple[dict, bool]:
    s = text.strip()
    # JSON?
    try:
        return json.loads(s), False
    except Exception:
        pass
    # YAML?
    if yaml is None:
        raise RuntimeError("Для YAML требуется пакет pyyaml")
    obj = yaml.safe_load(s)
    if not isinstance(obj, dict):
        raise RuntimeError("OpenAPI не распознан (ожидался объект)")
    return obj, True

# ===================== OPENAI =====================

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


def _build_messages(existing_spec_text: str,
                    prefer_yaml: bool,
                    ctx_json_text: str,
                    image_paths: List[Path],
                    req_text: str | None) -> list[dict]:
    user_content: list[dict] = [
        {"type": "text", "text": USER_INSTRUCTIONS},
        {"type": "text", "text": "ОБРАЗЕЦ СПЕЦИФИКАЦИИ (текущий OpenAPI):\n" + existing_spec_text},
        {"type": "text", "text": "CONTEXT (JSON):\n" + ctx_json_text},
        {"type": "text", "text": "ДИЗАЙН (скриншоты):"},
    ]
    for p in image_paths:
        user_content.append({"type": "image_url", "image_url": {"url": _img_as_data_url(p), "detail": "high"}})
    if req_text:
        user_content.append({"type": "text", "text": "⚠️ ЖЁСТКИЕ ОГРАНИЧЕНИЯ (REQUIREMENTS):\n" + req_text})
    fmt_hint = "YAML" if prefer_yaml else "JSON"
    user_content.append({"type": "text", "text": f"Верни фрагмент строго в формате {fmt_hint}."})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

# ===================== MERGE & OPERATION ID =====================

def _dict_merge_additive(base: dict, patch: dict, conflict_policy: str):
    """Аддитивный мердж: paths и подузлы components. Ничего не удаляем.
    При конфликте: skip/overwrite/fail.
    """
    def _merge_mapping(dst: dict, src: dict, level_tag: str):
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
            else:
                if isinstance(dst[k], dict) and isinstance(v, dict) and level_tag not in {"operation"}:
                    _merge_mapping(dst[k], v, level_tag)
                else:
                    if conflict_policy == "skip":
                        continue
                    elif conflict_policy == "overwrite":
                        dst[k] = v
                    else:
                        raise RuntimeError(f"Конфликт при мердже ключа '{k}' на уровне {level_tag}")

    # paths
    if "paths" in patch and isinstance(patch["paths"], dict):
        base.setdefault("paths", {})
        for path_key, ops in patch["paths"].items():
            if path_key not in base["paths"]:
                base["paths"][path_key] = ops
                continue
            if not isinstance(ops, dict):
                continue
            dst_ops = base["paths"][path_key] or {}
            for method_key, op_body in ops.items():
                if method_key not in dst_ops:
                    dst_ops[method_key] = op_body
                else:
                    if conflict_policy == "skip":
                        continue
                    elif conflict_policy == "overwrite":
                        dst_ops[method_key] = op_body
                    else:
                        raise RuntimeError(f"Конфликт: {path_key} {method_key} уже существует")

    # components
    if "components" in patch and isinstance(patch["components"], dict):
        base.setdefault("components", {})
        for node in ("schemas", "parameters", "responses", "securitySchemes", "requestBodies", "headers"):
            if node in patch["components"] and isinstance(patch["components"][node], dict):
                base["components"].setdefault(node, {})
                _merge_mapping(base["components"][node], patch["components"][node], f"components.{node}")

    # корневой security — добавим, если у базы нет
    if "security" in patch and "security" not in base:
        base["security"] = patch["security"]


def _sample_existing_operation_ids(base: dict, limit: int = 20) -> List[str]:
    out: List[str] = []
    for path, ops in (base.get("paths") or {}).items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if isinstance(op, dict) and "operationId" in op:
                out.append(str(op["operationId"]))
                if len(out) >= limit:
                    return out
    return out


def _detect_opid_style(samples: List[str]) -> str:
    """Грубая эвристика: camel|pascal|snake|kebab|lower.
    Возвращает строку-метку стиля.
    """
    if not samples:
        return "camel"
    s = samples[0]
    if "_" in s:
        return "snake"
    if "-" in s:
        return "kebab"
    if s[:1].isupper() and re.search(r"[A-Z][a-z]+[A-Z]", s):
        return "pascal"
    if re.search(r"[a-z][A-Z]", s):
        return "camel"
    if s.islower():
        return "lower"
    return "camel"


def _slug_segments_from_path(path: str) -> List[str]:
    segs = [seg for seg in path.strip("/").split("/") if seg]
    norm = []
    for seg in segs:
        seg = re.sub(r"[{}]", "", seg)  # {id} -> id
        seg = re.sub(r"[^A-Za-z0-9]+", "_", seg).strip("_")
        if seg:
            norm.append(seg.lower())
    return norm or ["root"]


def _to_camel(parts: List[str]) -> str:
    head = parts[0].lower()
    tail = [p.capitalize() for p in parts[1:]]
    return head + "".join(tail)


def _to_pascal(parts: List[str]) -> str:
    return "".join(p.capitalize() for p in parts)


def _to_snake(parts: List[str]) -> str:
    return "_".join(parts)


def _to_kebab(parts: List[str]) -> str:
    return "-".join(parts)


def _gen_opid(method: str, path: str, style: str) -> str:
    parts = [method.lower()] + _slug_segments_from_path(path)
    if style == "snake":
        return _to_snake(parts)
    if style == "kebab":
        return _to_kebab(parts)
    if style == "pascal":
        return _to_pascal(parts)
    if style == "lower":
        return "".join(parts)
    return _to_camel(parts)


def _apply_operation_id_policy(fragment: dict, base: dict, policy: str = "autogen") -> None:
    style = _detect_opid_style(_sample_existing_operation_ids(base))
    paths = fragment.get("paths") if isinstance(fragment, dict) else None
    if not isinstance(paths, dict):
        return
    for path, ops in paths.items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if not isinstance(op, dict):
                continue
            has = bool(op.get("operationId"))
            if has:
                continue
            if policy == "ignore":
                continue
            if policy == "require":
                raise RuntimeError(f"Отсутствует operationId у операции {method.upper()} {path}")
            # autogen
            op["operationId"] = _gen_opid(method, path, style)

# -------- x-stoplight helpers --------

def _get_stoplight_id(node: dict) -> str | None:
    if not isinstance(node, dict):
        return None
    x = node.get("x-stoplight")
    if isinstance(x, dict):
        v = x.get("id")
        return str(v) if v else None
    return None


def _set_stoplight_id(node: dict, new_id: str):
    node.setdefault("x-stoplight", {})
    if not isinstance(node["x-stoplight"], dict):
        node["x-stoplight"] = {}
    node["x-stoplight"]["id"] = new_id


def _collect_existing_stoplight_ids(spec: dict) -> set[str]:
    ids: set[str] = set()
    def _walk(o):
        if isinstance(o, dict):
            sid = _get_stoplight_id(o)
            if sid:
                ids.add(sid)
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
    _walk(spec)
    return ids


def _new_unique_id(taken: set[str]) -> str:
    while True:
        cand = uuid.uuid4().hex
        if cand not in taken:
            taken.add(cand)
            return cand


def _ensure_stoplight_ids_for_fragment(fragment: dict, base: dict, mode: str = "off"):
    """Проставляет x-stoplight.id для новых узлов фрагмента (operations, path items, components.*).
    mode: off|auto|require
    """
    if mode == "off":
        return
    taken = _collect_existing_stoplight_ids(base)

    def _touch(node: dict):
        sid = _get_stoplight_id(node)
        if sid:
            taken.add(sid)
            return
        _set_stoplight_id(node, _new_unique_id(taken))

    # Path items + operations
    paths = fragment.get("paths") if isinstance(fragment, dict) else None
    if isinstance(paths, dict):
        for path, item in paths.items():
            if isinstance(item, dict):
                _touch(item)  # path item
                for method, op in item.items():
                    if isinstance(op, dict) and method.lower() in {"get","post","put","delete","patch","options","head","trace"}:
                        _touch(op)

    # Components buckets
    comps = fragment.get("components") if isinstance(fragment, dict) else None
    if isinstance(comps, dict):
        for bucket in ("schemas","parameters","responses","requestBodies","headers","securitySchemes"):
            node = comps.get(bucket)
            if isinstance(node, dict):
                for name, obj in node.items():
                    if isinstance(obj, dict):
                        _touch(obj)

    if mode == "require":
        # лёгкая проверка — на интересных уровнях должен быть id
        def _check(o):
            if isinstance(o, dict):
                if ("summary" in o or "description" in o or "type" in o or ("in" in o and "name" in o)) and _get_stoplight_id(o) is None:
                    raise RuntimeError("REQUIRE: обнаружен узел без x-stoplight.id")
                for v in o.values():
                    _check(v)
            elif isinstance(o, list):
                for v in o:
                    _check(v)
        _check(fragment)

# -------- reporting helpers --------

def _iter_operations(spec: dict):
    """Итерирует операции как кортежи: (path, method, op_dict)."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for path, ops in paths.items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if isinstance(op, dict):
                yield path, method.lower(), op


def _summarize_changes(base: dict, fragment: dict, conflict_policy: str):
    """
    Возвращает два списка:
      - added_ops: операции, которых не было в базе (точно будут добавлены)
      - overwritten_ops: операции, которые заменим при policy='overwrite'
    Каждый элемент: dict(path=..., method=..., operationId=...)
    """
    base_keys = set()
    for path, method, _ in _iter_operations(base) or []:
        base_keys.add((path, method))

    added_ops, overwritten_ops = [], []
    for path, method, op in _iter_operations(fragment) or []:
        key = (path, method)
        info = {
            "path": path,
            "method": method.upper(),
            "operationId": op.get("operationId") or ""
        }
        if key in base_keys:
            if conflict_policy == "overwrite":
                overwritten_ops.append(info)
        else:
            added_ops.append(info)
    return added_ops, overwritten_ops

# ===================== MAIN =====================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("Augment existing OpenAPI with NEW methods (style-preserving)")
    ap.add_argument("--in-openapi", required=True, help="Существующий OpenAPI (yaml/json) — база и образец стиля")
    ap.add_argument("--out", required=True, help="Файл для записи результата (можно тот же, что и --in-openapi)")
    ap.add_argument("--model", required=True, help="LLM-модель (например, gpt-4o, gpt-4o-mini)")

    # ОБЯЗАТЕЛЬНЫЕ по требованиям пользователя
    ap.add_argument("--context", required=True, help="Доп. контекст (JSON) — ОБЯЗАТЕЛЕН")
    ap.add_argument("--design", action="append", default=[], help="Скриншот дизайна (можно несколько раз)")
    ap.add_argument("--design-dir", help="Папка со скриншотами дизайна")
    ap.add_argument("--design-recursive", action="store_true", help="Рекурсивный поиск изображений в --design-dir")

    # ОПЦИОНАЛЬНО
    ap.add_argument("--requirements", help="Текст/Markdown/JSON/YAML с жёсткими ограничениями (опционально)")

    ap.add_argument("--conflict-policy", choices=["skip", "overwrite", "fail"], default="skip")
    ap.add_argument("--opid-policy", choices=["autogen", "require", "ignore"], default="autogen",
                    help="Политика для operationId: autogen (по умолчанию) | require (ошибка, если нет) | ignore (ничего не делать)")
    ap.add_argument("--gen-stoplight", choices=["off", "auto", "require"], default="off",
                    help="Генерация x-stoplight.id для новых узлов: off (по умолчанию), auto, require")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Не вызывать LLM: сохранить подготовленные данные и выйти")

    args = ap.parse_args(argv)

    in_path = Path(args.in_openapi).resolve()
    if not in_path.exists():
        print(f"❌ Не найден исходный OpenAPI: {in_path}", file=sys.stderr)
        return 2

    # ВАЛИДАЦИЯ: обязателен контекст + хотя бы один источник дизайна
    ctx_path = Path(args.context).resolve()
    if not ctx_path.exists():
        print(f"❌ Не найден обязательный --context: {ctx_path}", file=sys.stderr)
        return 2

    design_paths = _collect_design_paths(args.design, args.design_dir, args.design_recursive)
    if not design_paths:
        print("❌ Требуется указать хотя бы один скриншот через --design ИЛИ папку через --design-dir", file=sys.stderr)
        return 2

    # читаем базовую спецификацию
    existing_text = _load_text_or_fail(in_path, "исходный OpenAPI")
    try:
        base_obj, base_is_yaml = _parse_openapi(existing_text)
    except Exception as e:
        print(f"❌ Не удалось распарсить исходный OpenAPI: {e}", file=sys.stderr)
        return 2

    # входные материалы
    ctx_json_text = _load_json_and_dump(ctx_path)
    req_text = None
    if args.requirements:
        req_text = _load_text_or_fail(Path(args.requirements).resolve(), "requirements")

    # подготовка сообщений
    messages = _build_messages(existing_text, base_is_yaml, ctx_json_text, design_paths, req_text)

    if args.dry_run:
        debug = {
            "system": SYSTEM_PROMPT,
            "user_messages": messages[1]["content"],
            "model": args.model,
            "temperature": args.temperature,
            "images_count": len(design_paths),
            "base_is_yaml": base_is_yaml,
        }
        Path(args.out).write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📝 DRY-RUN: подготовленные данные сохранены в {args.out}")
        return 0

    # вызов LLM
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        print("❌ Требуется пакет 'openai' (pip install openai)", file=sys.stderr)
        return 2

    try:
        client = _init_openai()
        resp = client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            messages=messages,
        )
        content = (resp.choices[0].message.content or "").strip()
        fragment_text_raw = _extract_code_block(content)
        fragment_text = _norm_to_format(fragment_text_raw, prefer_yaml=base_is_yaml)
        fragment_obj, _ = _parse_openapi(fragment_text)
    except Exception as e:
        fallback = str(Path(args.out).with_suffix(".augment_error.json"))
        debug = {
            "error": str(e),
            "system": SYSTEM_PROMPT,
            "user_messages": messages[1]["content"],
            "model": args.model,
            "temperature": args.temperature,
            "fragment_raw": content if 'content' in locals() else None
        }
        Path(fallback).write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"❌ Ошибка LLM/парсинга: {e}\nℹ️ Дамп сохранён: {fallback}", file=sys.stderr)
        return 1

    # политика operationId
    try:
        _apply_operation_id_policy(fragment_obj, base_obj, args.opid_policy)
    except Exception as e:
        print(f"❌ Проверка/генерация operationId: {e}", file=sys.stderr)
        return 1

    # x-stoplight (по запросу)
    try:
        _ensure_stoplight_ids_for_fragment(fragment_obj, base_obj, args.gen_stoplight)
    except Exception as e:
        print(f"❌ Генерация x-stoplight.id: {e}", file=sys.stderr)
        return 1

    # --- ПЛАНИРУЕМ ОТЧЁТ О ДОБАВЛЕННЫХ/ПЕРЕЗАПИСАННЫХ ОПЕРАЦИЯХ ---
    added_ops, overwritten_ops = _summarize_changes(base_obj, fragment_obj, args.conflict_policy)

    # мерджим фрагмент в базовую спецификацию
    try:
        _dict_merge_additive(base_obj, fragment_obj, args.conflict_policy)
    except Exception as e:
        print(f"❌ Ошибка мерджа: {e}", file=sys.stderr)
        return 1

    # --- ОТЧЁТ: какие операции добавлены/перезаписаны ---
    if added_ops:
        print("➕ Добавлены методы:")
        for it in added_ops:
            opid = f" (operationId: {it['operationId']})" if it["operationId"] else ""
            print(f"  - {it['method']} {it['path']}{opid}")
    else:
        print("ℹ️ Новых методов не добавлено.")

    if args.conflict_policy == "overwrite" and overwritten_ops:
        print("♻️ Перезаписаны существующие методы (policy=overwrite):")
        for it in overwritten_ops:
            opid = f" (operationId: {it['operationId']})" if it["operationId"] else ""
            print(f"  - {it['method']} {it['path']}{opid}")

    # сохраняем в первоначальном формате
    if base_is_yaml:
        if yaml is None:
            print("❌ Для сохранения YAML нужен pyyaml", file=sys.stderr)
            return 2
        out_text = yaml.safe_dump(base_obj, allow_unicode=True, sort_keys=False)
    else:
        out_text = json.dumps(base_obj, ensure_ascii=False, indent=2)

    out_path = Path(args.out).resolve()
    if args.in_place:
        out_path = in_path  # перезаписать оригинал

    out_path.write_text(out_text, encoding="utf-8")
    print(f"✅ Обновлённый OpenAPI сохранён в {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
