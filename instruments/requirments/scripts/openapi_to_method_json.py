#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/openapi_to_method_json.py — enum+description aware minimal extractor with deep $ref
Usage:
  python scripts/openapi_to_method_json.py --spec SPEC --method OPERATION_ID --minimal -o minimal.json
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
    if p.suffix.lower() == ".json" or data.lstrip().startswith("{"):
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
                # external $ref not handled to keep script compact
                return node
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

# ---------- helpers ----------
def normalize_id(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())

def enum_from_schema(sch: dict):
    if not isinstance(sch, dict):
        return None
    if isinstance(sch.get("enum"), list) and sch["enum"]:
        return sch["enum"]
    items = sch.get("items")
    if isinstance(items, dict) and isinstance(items.get("enum"), list) and items["enum"]:
        return items["enum"]
    return None

def desc_from_schema(sch: dict) -> Optional[str]:
    if isinstance(sch, dict) and isinstance(sch.get("description"), str) and sch["description"].strip():
        return sch["description"].strip()
    return None

# ---------- find operation ----------
def find_operation(doc: Dict[str, Any], method_name: str) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    target = normalize_id(method_name)
    paths = doc.get("paths", {}) or {}
    for path, item in paths.items():
        for m, op in (item or {}).items():
            if m.lower() not in {"get","post","put","patch","delete","head","options","trace"}: continue
            if isinstance(op, dict) and normalize_id(op.get("operationId")) == target:
                return m.upper(), path, op, item
    for path, item in paths.items():
        for m, op in (item or {}).items():
            if m.lower() not in {"get","post","put","patch","delete","head","options","trace"}: continue
            if isinstance(op, dict) and normalize_id(op.get("summary")) == target:
                return m.upper(), path, op, item
    for path, item in paths.items():
        for m, op in (item or {}).items():
            if m.lower() not in {"get","post","put","patch","delete","head","options","trace"}: continue
            if not isinstance(op, dict): continue
            tags = op.get("tags") or []
            cand = f"{(tags[0] if tags else '').upper()}.{m.upper()}"
            if normalize_id(cand) == target:
                return m.upper(), path, op, item
    raise KeyError(f"Метод '{method_name}' не найден.")

# ---------- flatten ----------
def _flatten_schema(schema: Dict[str, Any],
                    required: Optional[List[str]] = None,
                    prefix: str = "",
                    resolver: Optional[RefResolver] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(schema, dict):
        return out

    # composition
    for comb in ("allOf","oneOf","anyOf"):
        if comb in schema and isinstance(schema[comb], list):
            for sub in schema[comb]:
                out.extend(_flatten_schema(sub, required=required, prefix=prefix, resolver=resolver))

    # local deref
    if "$ref" in schema and resolver is not None:
        try:
            schema = resolver.deref(schema)
        except Exception:
            pass

    t = (schema.get("type") or "").lower()
    req = set(required or [])

    # enum leaf (with description)
    if "enum" in schema and "properties" not in schema and "items" not in schema:
        out.append({
            "path": prefix.rstrip(".[]"),
            "type": t or "string",
            "required": (prefix.split(".")[-1] in req) if prefix else False,
            "enum": schema.get("enum"),
            "description": desc_from_schema(schema)
        })

    # object
    if t == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        child_req = set(schema.get("required") or [])
        for name, sub in props.items():
            node = sub
            if isinstance(node, dict) and "$ref" in node and resolver is not None:
                try:
                    node = resolver.deref(node)
                except Exception:
                    pass
            path = f"{prefix}.{name}" if prefix else name
            ntype = (node.get("type") or "object") if isinstance(node, dict) else "object"
            item = {"path": path, "type": ntype, "required": name in child_req}
            if isinstance(node, dict):
                ev = enum_from_schema(node)
                if ev is not None:
                    item["enum"] = ev
                dsc = desc_from_schema(node)
                if dsc is not None:
                    item["description"] = dsc
            out.append(item)
            if isinstance(node, dict):
                nt = (node.get("type") or "").lower()
                if nt == "object" or "properties" in node:
                    out.extend(_flatten_schema(node, list(child_req), path, resolver))
                elif nt == "array":
                    items = node.get("items") or {}
                    if isinstance(items, dict) and "$ref" in items and resolver is not None:
                        try:
                            items = resolver.deref(items)
                        except Exception:
                            pass
                    it = (items.get("type") or "object") if isinstance(items, dict) else "object"
                    arr = {"path": f"{path}[]", "type": f"array<{it}>", "required": False}
                    ev = enum_from_schema(items) if isinstance(items, dict) else None
                    if ev is not None:
                        arr["enum"] = ev
                    dsc = desc_from_schema(items) if isinstance(items, dict) else None
                    if dsc is None:
                        dsc = desc_from_schema(node) if isinstance(node, dict) else None
                    if dsc is not None:
                        arr["description"] = dsc
                    out.append(arr)
                    if isinstance(items, dict) and (it == "object" or "properties" in items):
                        out.extend(_flatten_schema(items, items.get("required"), f"{path}[]", resolver))

    # array root
    if t == "array":
        items = schema.get("items") or {}
        if isinstance(items, dict) and "$ref" in items and resolver is not None:
            try:
                items = resolver.deref(items)
            except Exception:
                pass
        it = (items.get("type") or "object") if isinstance(items, dict) else "object"
        arr = {"path": f"{prefix}[]" if prefix else "[]", "type": f"array<{it}>", "required": False}
        ev = enum_from_schema(items) if isinstance(items, dict) else None
        if ev is not None:
            arr["enum"] = ev
        dsc = desc_from_schema(items) if isinstance(items, dict) else None
        if dsc is None:
            dsc = desc_from_schema(schema)
        if dsc is not None:
            arr["description"] = dsc
        out.append(arr)
        if isinstance(items, dict) and (it == "object" or "properties" in items):
            out.extend(_flatten_schema(items, items.get("required"), arr["path"], resolver))

    # primitive leaf w/o enum
    if t in {"string","number","integer","boolean"} and "properties" not in schema and "items" not in schema:
        if prefix:
            item = {"path": prefix, "type": t, "required": (prefix.split(".")[-1] in req)}
            dsc = desc_from_schema(schema)
            if dsc is not None:
                item["description"] = dsc
            out.append(item)

    # dedupe
    ded: Dict[str, Dict[str, Any]] = {}
    for p in out:
        ded[p["path"]] = p
    return list(ded.values())

# ---------- build ----------
def build_method_json(doc: Dict[str, Any], method_name: str, base_dir: Union[str, Path]) -> Dict[str, Any]:
    http_method, path, op_raw, path_item_raw = find_operation(doc, method_name)
    resolver = RefResolver(doc, base_dir=base_dir)
    path_item = resolver.deref(path_item_raw) if isinstance(path_item_raw, dict) else {}
    op = resolver.deref(op_raw) if isinstance(op_raw, dict) else {}

    op_id = op.get("operationId") or method_name
    name = op.get("summary") or op_id
    summary = op.get("description") or op.get("summary") or ""

    # collect query/path params (store schema + description)
    params_out = []
    all_params = []
    for lvl in (path_item, op):
        items = (lvl.get("parameters") or []) if isinstance(lvl, dict) else []
        if isinstance(items, list): all_params.extend(items)
    seen = set()
    for p in all_params:
        if not isinstance(p, dict): continue
        key = (p.get("name"), p.get("in"))
        if key in seen: continue
        seen.add(key)
        loc = (p.get("in") or "").lower()
        sch = p.get("schema") or {}
        if isinstance(sch, dict):
            sch = resolver.deref(sch)
        desc = p.get("description") or (sch.get("description") if isinstance(sch, dict) else None)
        item = {
            "name": p.get("name"), "in": loc, "required": bool(p.get("required", False)),
            "schema": sch if isinstance(sch, dict) else None
        }
        if desc:
            item["description"] = desc
        if loc in {"query","path"}:
            params_out.append(item)

    # request body
    body = {"required": False}
    rb = op.get("requestBody")
    if isinstance(rb, dict):
        content = rb.get("content") or {}
        ct = "application/json" if "application/json" in content else (sorted(content.keys())[0] if content else None)
        body["required"] = bool(rb.get("required", False))
        if ct:
            entry = content.get(ct, {})
            schema = entry.get("schema")
            if isinstance(schema, dict):
                schema = resolver.deref(schema)
            body["schema"] = schema if isinstance(schema, dict) else None
            bdesc = entry.get("description") or (schema.get("description") if isinstance(schema, dict) else None)
            if bdesc:
                body["description"] = bdesc

    # responses
    responses = []
    raw_resps = op.get("responses") or {}
    for status, r in raw_resps.items():
        if not isinstance(r, dict):
            responses.append({"status": status, "description": "", "schema": None})
            continue
        r = resolver.deref(r)
        content = r.get("content") or {}
        ct = "application/json" if "application/json" in content else (sorted(content.keys())[0] if content else None)
        schema_any = None
        if ct:
            entry = content[ct]
            schema_any = entry.get("schema")
            if isinstance(schema_any, dict):
                schema_any = resolver.deref(schema_any)
        responses.append({
            "status": int(status) if str(status).isdigit() else status,
            "description": r.get("description") or "",
            "schema": schema_any if isinstance(schema_any, dict) else None
        })

    return {
        "id": str(op_id),
        "name": str(name),
        "summary": str(summary),
        "http_method": http_method,
        "endpoint": path,
        "params": params_out,
        "body": body,
        "responses": responses
    }

def build_minimal_slice(method_obj: dict, spec: Dict[str, Any], base_dir: Union[str, Path]) -> dict:
    resolver = RefResolver(spec, base_dir=base_dir)

    # query
    q = []
    for p in method_obj.get("params", []) or []:
        if (p.get("in") or "").lower() == "query":
            item = {
                "name": p.get("name"),
                "required": bool(p.get("required", False))
            }
            if p.get("description"):
                item["description"] = p["description"]
            sch = p.get("schema") or {}
            ev = enum_from_schema(sch) if isinstance(sch, dict) else None
            if ev is not None:
                item["enum"] = ev
            q.append(item)

    # path params
    path_params = []
    for p in method_obj.get("params", []) or []:
        if (p.get("in") or "").lower() == "path":
            item = {
                "name": p.get("name"),
                "required": bool(p.get("required", False))
            }
            if p.get("description"):
                item["description"] = p["description"]
            sch = p.get("schema") or {}
            ev = enum_from_schema(sch) if isinstance(sch, dict) else None
            if ev is not None:
                item["enum"] = ev
            path_params.append(item)

    # body params
    body_block = {"required": False, "params": []}
    body = method_obj.get("body") or {}
    body_block["required"] = bool(body.get("required", False))
    if body.get("description"):
        body_block["description"] = body["description"]
    sch = body.get("schema")
    if isinstance(sch, dict):
        body_block["params"] = _flatten_schema(sch, required=sch.get("required"), resolver=resolver)

    # responses
    resps = []
    for r in method_obj.get("responses", []) or []:
        sch = r.get("schema")
        params = _flatten_schema(sch, required=(sch or {}).get("required"), resolver=resolver) if isinstance(sch, dict) else []
        resps.append({
            "code": r.get("status") or r.get("code"),
            "description": r.get("description") or "",
            "params": params
        })

    return {
        "method_name": method_obj.get("name") or method_obj.get("id"),
        "http_method": method_obj.get("http_method"),
        "description": method_obj.get("summary") or "",
        "request": {
            "path": method_obj.get("endpoint"),
            "path_params": path_params,
            "query": q,
            "body": body_block
        },
        "responses": resps
    }

# ---------- CLI ----------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("OpenAPI → minimal with enum+description")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("-o", "--output")
    ap.add_argument("--minimal", action="store_true")
    args = ap.parse_args(argv)

    spec_path = Path(args.spec).resolve()
    spec = load_spec(spec_path)
    method_json = build_method_json(spec, args.method, base_dir=spec_path.parent)
    if args.minimal:
        minimal = build_minimal_slice(method_json, spec, base_dir=spec_path.parent)
        save_json(minimal, args.output)
    else:
        save_json(method_json, args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
