#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
openapi_to_method_json.py — deep $ref + protobuf-эвристики

Использование:
  python scripts/openapi_to_method_json.py --spec spec/pro-openapi.yaml --method getPersonalNews --minimal -o minimal.json

Зависимости: pyyaml (pip install pyyaml)
"""
from __future__ import annotations
import argparse, json, re, sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# ---------- IO ----------
def load_spec(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    data = p.read_text(encoding="utf-8")
    is_json_ext = p.suffix.lower() == ".json"
    if is_json_ext or data.lstrip().startswith("{"):
        return json.loads(data)
    if yaml is None:
        raise RuntimeError("Файл похож на YAML. Установите pyyaml: pip install pyyaml")
    return yaml.safe_load(data)

def save_json(obj: Any, path: Optional[Union[str, Path]]) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    if path:
        Path(path).write_text(payload, encoding="utf-8")
    else:
        print(payload)

# ---------- Deep $ref resolver ----------
class RefResolver:
    def __init__(self, root_doc: Dict[str, Any], base_dir: Union[str, Path]):
        self.root_doc = root_doc
        self.base_dir = Path(base_dir).resolve()
        self._doc_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _unescape_pointer_token(token: str) -> str:
        return token.replace("~1", "/").replace("~0", "~")

    def _resolve_pointer(self, doc: Any, pointer: str) -> Any:
        if pointer in ("#", ""):
            return doc
        if not pointer.startswith("#/"):
            raise KeyError(f"Unsupported JSON Pointer: {pointer}")
        cur = doc
        for raw in pointer[2:].split("/"):
            key = self._unescape_pointer_token(raw)
            if isinstance(cur, list):
                cur = cur[int(key)]
            elif isinstance(cur, dict):
                cur = cur[key]
            else:
                raise KeyError(f"Cannot traverse at '{key}'")
        return cur

    def _load_external_doc(self, ref_path: str) -> Dict[str, Any]:
        abs_path = str((self.base_dir / ref_path).resolve())
        if abs_path in self._doc_cache:
            return self._doc_cache[abs_path]
        p = Path(abs_path)
        if not p.exists():
            raise FileNotFoundError(f"External $ref not found: {abs_path}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json" or text.lstrip().startswith("{"):
            doc = json.loads(text)
        else:
            if yaml is None:
                raise RuntimeError("Нужен pyyaml для YAML.")
            doc = yaml.safe_load(text)
        self._doc_cache[abs_path] = doc
        return doc

    def deref(self, node: Any, seen: Optional[set[str]] = None) -> Any:
        if seen is None: seen = set()
        if isinstance(node, list):
            return [self.deref(it, seen) for it in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node and isinstance(node["$ref"], str):
            ref_str = node["$ref"]
            target_doc, pointer = self.root_doc, "#"
            if ref_str.startswith("#"):
                pointer = ref_str
                ref_key = f"{self.base_dir}/(root){pointer}"
            else:
                if "://" in ref_str:
                    return node  # не тянем HTTP
                if "#/" in ref_str:
                    file_part, ptr = ref_str.split("#", 1)
                    pointer = "#" + ptr
                else:
                    file_part, pointer = ref_str, "#"
                target_doc = self._load_external_doc(file_part)
                ref_key = f"{str((self.base_dir / file_part).resolve())}{pointer}"
            if ref_key in seen:
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return self.deref(merged, seen)
            try:
                target = self._resolve_pointer(target_doc, pointer)
            except Exception:
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return self.deref(merged, seen)
            merged = deepcopy(target) if isinstance(target, (dict, list)) else target
            if isinstance(merged, dict):
                for k, v in node.items():
                    if k != "$ref":
                        merged[k] = v
            seen_next = set(seen); seen_next.add(ref_key)
            return self.deref(merged, seen_next)
        out = {}
        for k, v in node.items():
            out[k] = self.deref(v, seen)
        return out

# ---------- Поиск операции ----------
def normalize_id(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())

def find_operation(doc: Dict[str, Any], method_name: str) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    target = normalize_id(method_name)
    paths = doc.get("paths", {}) or {}
    # operationId
    for path, item in paths.items():
        for m, op in (item or {}).items():
            if m.lower() not in {"get","post","put","patch","delete","head","options","trace"}: continue
            if isinstance(op, dict) and normalize_id(op.get("operationId")) == target:
                return m.upper(), path, op, item
    # summary
    for path, item in paths.items():
        for m, op in (item or {}).items():
            if m.lower() not in {"get","post","put","patch","delete","head","options","trace"}: continue
            if isinstance(op, dict) and normalize_id(op.get("summary")) == target:
                return m.upper(), path, op, item
    # heuristic TAG.METHOD
    for path, item in paths.items():
        for m, op in (item or {}).items():
            if m.lower() not in {"get","post","put","patch","delete","head","options","trace"}: continue
            if not isinstance(op, dict): continue
            tags = op.get("tags") or []
            cand = f"{(tags[0] if tags else '').upper()}.{m.upper()}"
            if normalize_id(cand) == target:
                return m.upper(), path, op, item
    raise KeyError(f"Метод '{method_name}' не найден.")

# ---------- Эвристики типов/пагинации/сортировки/фильтров ----------
PAGINATION_OFFS = {"limit", "offset"}
PAGINATION_PAGE = {"page", "page_size", "pagesize", "per_page", "perpage"}
PAGINATION_CURSOR = {"cursor", "page_token", "next_token", "starting_after", "ending_before"}
SORT_NAMES = {"sort", "order", "orderby", "ordering", "sort_by", "sortby", "order_by"}

def param_type_from_schema(s: Dict[str, Any]) -> Optional[str]:
    t = (s or {}).get("type")
    if isinstance(t, str) and t.lower() in {"string","number","integer","boolean"}:
        return t.lower()
    fmt = (s or {}).get("format", "")
    if fmt in {"date","date-time"}:
        return "datetime" if fmt == "date-time" else "date"
    return None

def classify_pagination(qparams: List[Dict[str, Any]]) -> str:
    names = {(p.get("name") or "").lower() for p in qparams}
    if {"limit","offset"}.issubset(names): return "offset"
    if ("page" in names) and ({"page_size","pagesize","per_page","perpage"} & names): return "page"
    if names & PAGINATION_CURSOR: return "cursor"
    return "none"

def extract_sorting(qparams: List[Dict[str, Any]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    sort_param = next((p for p in qparams if (p.get("name","").lower() in SORT_NAMES)), None)
    if sort_param:
        res["fields"] = []
        sch = sort_param.get("schema") or {}
        enum_vals = sch.get("enum")
        if isinstance(enum_vals, list) and enum_vals:
            fields = {str(v).split(":")[0] for v in enum_vals if isinstance(v, str)}
            res["fields"] = sorted(fields) if fields else enum_vals
        default = sch.get("default")
        if isinstance(default, str) and default:
            res["default"] = default
        elif isinstance(sort_param.get("example"), str):
            res["default"] = sort_param["example"]
    return res

def extract_filtering(qparams: List[Dict[str, Any]]) -> List[str]:
    excluded = PAGINATION_OFFS | PAGINATION_PAGE | PAGINATION_CURSOR | SORT_NAMES
    seen, out = set(), []
    for p in qparams:
        name = (p.get("name") or "").lower()
        if name and name not in excluded:
            orig = p.get("name")
            if orig not in seen:
                out.append(orig); seen.add(orig)
    return out

# ---------- Protobuf эвристики ----------
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", (s or "")).lower()

def _base_from_method(method_name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "", method_name or "")
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", s)
    if not parts: return s
    verbs = {"get","list","create","update","delete","put","post","patch","mark","set","send","fetch","remove"}
    return "".join(parts[1:]) if parts[0].lower() in verbs and len(parts) >= 2 else "".join(parts)

def _guess_component_name(schemas: Dict[str, Any], base: str, kind: str) -> Optional[str]:
    base_slug = _slug(base)
    def score(name: str) -> int:
        nslug = _slug(name); sc = 0
        if base_slug and base_slug in nslug: sc += 5
        if kind == "request" and "request" in nslug: sc += 3
        if kind == "response" and "response" in nslug: sc += 3
        if "protobufdto" in nslug: sc += 1
        # сильные совпадения
        targets = [f"{base}RequestProtobufDTO", f"{base}RequestDTO", f"{base}Request",
                   f"{base}ResponseProtobufDTO", f"{base}ResponseDTO", f"{base}Response", f"{base}ProtobufDTO"]
        if _slug(name) in {_slug(t) for t in targets}: sc += 10
        return sc
    best, best_sc = None, -1
    for n in schemas.keys():
        sc = score(n)
        if sc > best_sc:
            best, best_sc = n, sc
    if best_sc >= 5:
        return best
    # доп. попытка — по description
    for n, obj in schemas.items():
        desc = (obj or {}).get("description") or ""
        if base and base.lower() in desc.lower():
            if kind == "request" and "request" in desc.lower():
                return n
            if kind == "response" and "response" in desc.lower():
                return n
    return None

def _prefer_protobuf_schema_if_empty(media_type: str, schema: Any, components: Dict[str, Any], method_name: str, resolver: RefResolver, kind: str) -> Any:
    """
    Если content-type = application/x-protobuf и schema пустая/примитивная строка,
    пробуем найти components.schemas[*Request/Response*] по имени метода.
    """
    if str(media_type).lower() != "application/x-protobuf":
        return schema
    # "пустая" схема — string или None
    is_trivial = (not isinstance(schema, dict)) or (schema.get("type") == "string" and len(schema.keys()) <= 2)
    if not is_trivial:
        return schema
    comps = (components or {}).get("schemas") or {}
    base = _base_from_method(method_name)
    target_name = _guess_component_name(comps, base, kind=kind)
    if not target_name:
        return schema
    cand = comps.get(target_name)
    if not isinstance(cand, dict):
        return schema
    return resolver.deref(cand)

# ---------- Главная сборка ----------
def build_method_json(doc: Dict[str, Any], method_name: str, base_dir: Union[str, Path]) -> Dict[str, Any]:
    http_method, path, op_raw, path_item_raw = find_operation(doc, method_name)
    resolver = RefResolver(doc, base_dir=base_dir)
    path_item = resolver.deref(path_item_raw) if isinstance(path_item_raw, dict) else {}
    op = resolver.deref(op_raw) if isinstance(op_raw, dict) else {}

    op_id = op.get("operationId") or method_name
    name = op.get("summary") or op_id
    summary = op.get("description") or op.get("summary") or ""

    # параметры
    all_params = []
    for level in (path_item, op):
        items = (level.get("parameters") or []) if isinstance(level, dict) else []
        if isinstance(items, list):
            all_params.extend(items)
    seen, uniq = set(), []
    for p in all_params:
        if not isinstance(p, dict): continue
        key = (p.get("name"), p.get("in"))
        if key not in seen:
            uniq.append(p); seen.add(key)
    params_out, headers_out = [], []
    for p in uniq:
        loc = (p.get("in") or "").lower()
        sch = p.get("schema") or {}
        p_out = {
            "name": p.get("name"),
            "in": loc,
            "type": param_type_from_schema(sch) if isinstance(sch, dict) else None,
            "required": bool(p.get("required", False)),
        }
        if "example" in p: p_out["example"] = p["example"]
        elif isinstance(sch, dict) and "default" in sch: p_out["example"] = sch["default"]
        if loc in {"query","path"}: params_out.append(p_out)
        elif loc == "header":
            headers_out.append({"name": p.get("name"), "in": "header", "required": p_out["required"], "schema": sch if isinstance(sch, dict) else None})

    # requestBody
    body = {"required": False}
    rb = op.get("requestBody")
    if isinstance(rb, dict):
        content = rb.get("content") or {}
        # приоритет JSON → иначе берём первый
        ct = None
        for cand in ("application/json", "application/*+json"):
            if cand in content: ct = cand; break
        if ct is None and content: ct = sorted(content.keys())[0]
        body["required"] = bool(rb.get("required", False))
        if ct:
            entry = content.get(ct, {})
            schema = entry.get("schema")
            if isinstance(schema, dict): schema = resolver.deref(schema)
            # protobuf-фикс, если schema пустая
            schema = _prefer_protobuf_schema_if_empty(ct, schema, (doc.get("components") or {}), op_id or name, resolver, kind="request")
            example = None
            if "example" in entry: example = entry["example"]
            elif "examples" in entry and isinstance(entry["examples"], dict):
                for v in entry["examples"].values():
                    if isinstance(v, dict) and "value" in v: example = v["value"]; break
            body.update({"schema": schema if isinstance(schema, dict) else {"type": "string"} if schema is None else schema,
                         "example": example})
    if headers_out:
        body = {**body, "headers": headers_out}

    # auth
    def detect_auth(doc: Dict[str, Any], op: Dict[str, Any]) -> str:
        def map_scheme(sch: Dict[str, Any]) -> str:
            t = (sch.get("type") or "").lower()
            if t == "http":
                scheme = (sch.get("scheme") or "").lower()
                if scheme == "bearer": return "bearer"
                if scheme == "basic":  return "none"
            if t == "apikey": return "apiKey"
            if t == "oauth2": return "bearer"
            return "none"
        security = op.get("security", None)
        if security is None: security = doc.get("security", [])
        if not security: return "none"
        comps = ((doc.get("components") or {}).get("securitySchemes") or {})
        if not isinstance(comps, dict): return "none"
        for sec in security:
            if not isinstance(sec, dict): continue
            for scheme_name in sec.keys():
                scheme = comps.get(scheme_name)
                if isinstance(scheme, dict):
                    scheme = resolver.deref(scheme)
                    return map_scheme(scheme)
        return "none"

    auth = detect_auth(doc, op)

    # пагинация/сортировка/фильтрация
    qparams = [p for p in params_out if p.get("in") == "query"]
    pagination = classify_pagination(qparams)
    sorting = extract_sorting(qparams)
    filtering = extract_filtering(qparams)

    # responses (включая schema)
    responses_out = []
    raw_resps = op.get("responses") or {}
    for status, r in raw_resps.items():
        if not isinstance(r, dict):
            responses_out.append({"status": int(status) if str(status).isdigit() else status, "description": ""})
            continue
        r = resolver.deref(r)
        desc = r.get("description") or ""
        content = r.get("content") or {}
        example, schema_any = None, None

        def pick_example(entry: Dict[str, Any]) -> Any:
            if "example" in entry: return entry["example"]
            if "examples" in entry and isinstance(entry["examples"], dict):
                for v in entry["examples"].values():
                    if isinstance(v, dict) and "value" in v: return v["value"]
            return None

        # JSON предпочтительнее
        ct = None
        if "application/json" in content:
            ct = "application/json"
        elif content:
            ct = sorted(content.keys())[0]

        if ct:
            entry = content[ct]
            schema_any = entry.get("schema")
            if isinstance(schema_any, dict):
                schema_any = resolver.deref(schema_any)
            # protobuf-фикс
            schema_any = _prefer_protobuf_schema_if_empty(ct, schema_any, (doc.get("components") or {}), op_id or name, resolver, kind="response")
            example = pick_example(entry)

        responses_out.append({
            "status": int(status) if str(status).isdigit() else status,
            "description": desc,
            "schema": schema_any if isinstance(schema_any, dict) else None,
            "example": example
        })

    notes = []
    if op.get("deprecated"): notes.append("Метод помечен как deprecated.")

    return {
        "id": str(op_id),
        "name": str(name),
        "summary": str(summary),
        "http_method": http_method,
        "endpoint": path,
        "auth": auth,
        "params": params_out or [],
        "body": body,
        "pagination": pagination,
        "sorting": sorting or {},
        "filtering": filtering or [],
        "responses": responses_out or [],
        "notes": "; ".join(notes)
    }



# ---------- Minimal slice (flattened, relevant-only) ----------
def _collect_schema_fields(schema, required=None, prefix=""):
    out = []
    req_set = set(required or [])
    if not isinstance(schema, dict):
        return out

    # handle composition
    for comb in ("allOf", "oneOf", "anyOf"):
        if comb in schema and isinstance(schema[comb], list):
            for sub in schema[comb]:
                out.extend(_collect_schema_fields(sub, required=required, prefix=prefix))

    t = schema.get("type") if isinstance(schema.get("type"), str) else None

    # enum leaf
    if "enum" in schema and not schema.get("properties") and t in (None, "string", "number", "integer", "boolean"):
        out.append({
            "path": prefix.rstrip(".[]"),
            "required": (prefix.split(".")[-1] in req_set) if prefix else False,
            "enum": schema.get("enum")
        })

    # object
    if t == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        child_req = set(schema.get("required") or [])
        for name, sub in props.items():
            pfx = f"{prefix}.{name}" if prefix else name
            sub_out = _collect_schema_fields(sub, required=list(child_req), prefix=pfx)
            if sub_out:
                out.extend(sub_out)
            else:
                item = {"path": pfx, "required": name in child_req}
                if isinstance(sub, dict) and "enum" in sub:
                    item["enum"] = sub["enum"]
                out.append(item)

    # array
    if t == "array" and "items" in schema:
        ap = f"{prefix}[]" if prefix else "[]"
        sub_out = _collect_schema_fields(schema["items"], required=None, prefix=ap)
        out.extend(sub_out if sub_out else [{"path": ap, "required": False}])

    # primitive leaf
    if t in {"string","number","integer","boolean"} and "properties" not in schema and "items" not in schema:
        if prefix:
            out.append({"path": prefix, "required": (prefix.split(".")[-1] in req_set) if prefix else False})

    # dedupe by path
    ded = {}
    for p in out:
        ded[p["path"]] = p
    return list(ded.values())


def build_minimal_slice(method_obj: dict) -> dict:
    """Convert rich method_json produced by this script to a flattened minimal slice."""
    # Query params
    q = []
    for p in method_obj.get("params", []) or []:
        if (p.get("in") or "").lower() == "query":
            item = {"name": p.get("name"), "required": bool(p.get("required", False))}
            sch = p.get("schema") or {}
            if isinstance(sch, dict) and "enum" in sch:
                item["enum"] = sch["enum"]
            q.append(item)

    # Body params
    body_block = {"required": False, "params": []}
    body = method_obj.get("body") or {}
    body_block["required"] = bool(body.get("required", False))
    sch = body.get("schema")
    if isinstance(sch, dict):
        body_block["params"] = _collect_schema_fields(sch, required=sch.get("required"))

    # Responses flattened
    resps = []
    for r in method_obj.get("responses", []) or []:
        if not isinstance(r, dict): 
            continue
        sch = r.get("schema")
        params = _collect_schema_fields(sch, required=(sch or {}).get("required")) if isinstance(sch, dict) else []
        resps.append({
            "code": r.get("status") or r.get("code"),
            "description": r.get("description") or "",
            "params": params,
            "example": r.get("example")
        })

    return {
        "method_name": method_obj.get("name") or method_obj.get("summary") or method_obj.get("id"),
        "http_method": method_obj.get("http_method"),
        "description": method_obj.get("summary") or "",
        "request": {
            "path": method_obj.get("endpoint"),
            "query": q,
            "body": body_block
        },
        "responses": resps
    }

# ---------- CLI ----------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("Convert OpenAPI method → simple JSON (deep $ref + protobuf heuristics) + minimal slice")
    ap.add_argument("--spec", required=True, help="Путь к OpenAPI (json/yaml)")
    ap.add_argument("--method", required=True, help="Название метода (обычно operationId, можно summary)")
    ap.add_argument("-o", "--output", help="Куда сохранить JSON (если не задано — stdout)")
    ap.add_argument("--minimal", action="store_true", help="Вывести только минимальный срез (flattened)")
    ap.add_argument("--with-minimal", action="store_true", help="Добавить ключ 'minimal' с плоским срезом")
    args = ap.parse_args(argv)
    try:
        spec_path = Path(args.spec).resolve()
        spec = load_spec(spec_path)
        method_json = build_method_json(spec, args.method, base_dir=spec_path.parent)
        if args.minimal:
            save_json(build_minimal_slice(method_json), args.output)
        elif args.with_minimal:
            obj = dict(method_json)
            obj["minimal"] = build_minimal_slice(method_json)
            save_json(obj, args.output)
        else:
            save_json(method_json, args.output)
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
