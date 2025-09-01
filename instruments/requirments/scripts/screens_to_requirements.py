"""
extract_requirements_v4.py
--------------------------
Принимает папку со скриншотами, извлекает требования и сохраняет:
  - requirements.json — только факты из макетов
  - design.json — допущения/вопросы/рекомендации

Зависимости:
  pip install openai
  export OPENAI_API_KEY=...

Запуск:
  python scripts/screens_to_requirements.py ./screens --out requirements.json --design design.json
"""

import argparse, json, os, sys, base64
from typing import List, Dict, Any

def build_prompt() -> str:
    return (
        "Ты аналитик. По изображениям зафиксируй ТОЛЬКО проверяемые функциональные "
        "требования для фронтенд-разработки. "
        "Никаких конкретных текстов, дат, сумм или чисел. "
        "Все результаты раздели на два блока:\n"
        "  - requirements: список экранов. Для каждого экрана укажи:\n"
        "      • screen (название)\n"
        "      • states (ТОЛЬКО те, что явно показаны в макете)\n"
        "      • elements (структурные элементы: type, fields, grouping, sorting)\n"
        "      • actions (какие действия доступны)\n"
        "      • navigation (откуда и куда есть переходы)\n"
        "      • constraints (только те ограничения, что очевидны из макета)\n"
        "  - design: всё остальное. Сюда выноси замечания, вопросы и рекомендации по улучшению дизайна "
        "(например: добавить состояния ошибки, предусмотреть индикатор загрузки, уточнить обработку длинных текстов).\n"
        "Верни один JSON-объект с двумя ключами: requirements и design. "
        "requirements попадёт в requirements.json, design — в design.json."
    )




def call_model(image_paths: List[str], model: str = "gpt-4o") -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI()

    # Собираем контент с изображениями (data URL base64)
    content = [{"type": "text", "text": build_prompt()}]
    for p in image_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if p.lower().endswith("png") else "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })

    # Совместимый вызов
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Отвечай только корректным JSON без пояснений."},
            {"role": "user", "content": content},
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    # Пытаемся распарсить JSON; если вокруг текст — вырезаем блоки скобок.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def collect_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg")
    files = []
    for root, _, names in os.walk(folder):
        for n in sorted(names):
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    return files

def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Папка со скриншотами")
    parser.add_argument("--out", default="requirements.json")
    parser.add_argument("--design", default="design.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args(argv)

    files = collect_images(args.folder)
    if not files:
        print("Изображения не найдены.")
        sys.exit(1)

    result = call_model(files, model=args.model)
    requirements = result.get("requirements", [])
    design = result.get("design", {"assumptions": [], "recommendations": [], "questions": []})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(requirements, f, ensure_ascii=False, indent=2)
    with open(args.design, "w", encoding="utf-8") as f:
        json.dump(design, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} and {args.design}")

if __name__ == "__main__":
    main(sys.argv[1:])
