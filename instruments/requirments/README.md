# Инструкция по работе со скриптами и подготовке артефактов

## Схема работы со скриптами

Общий процесс состоит из нескольких шагов:  
1. Подготовка данных.  
2. Формирование артефакта.  
3. Аналитическая проверка и финализация.  

> ⚠️ Для работы с LLM необходимо заранее иметь **API-ключ** от выбранной модели (пока что только OpenAI).  
> Ключ должен хранится в переменной окружения`OPENAI_API_KEY`

**Linux / macOS**
```bash
export OPENAI_API_KEY="ваш_ключ"
```
**Windows PowerShell**
```bash
setx OPENAI_API_KEY "ваш_ключ"
```

## 0. Предподготовка
- Положить примеры артефактов в формате pdf (2-3 штуки) в директорию `docs/pdf`
- Положить OpenAPI файлы со спекой проекта (*опционально*)
- Установить переменную окружения `OPENAI_API_KEY`
- Установить зависимости из файла requirments.txt

---

## 1. Подготовка данных

### `pdf2md.py`  
- Преобразует PDF в Markdown.  
- Нужно для того, чтобы LLM проще воспринимала текстовые данные, а не PDF-документы.  

**Пример запуска:**
```bash
 python scripts/pdf2md.py \
  --example utils/example.md \
  --section "%Название подраздела страницы, который необходимо извлечь%" \
  --model gpt-4o \
  docs/pdf \
  --out-dir "docs/md"
```

### `extract_style_profile.py`  
- Получает **общий профиль стиля** на основе примеров артефактов.  
- Извлекает **few-shot примеры** из артефактов.  
- Цель: чтобы LLM понимала, в каком стиле и формате необходимо генерировать новый артефакт. 

**Пример запуска:**
```bash
python scripts/extract_style_profile.py \
  "docs/md/*.md" \
  -o style/style_profile.json \
  --org "My Company" \
  --lang ru-RU \
  -m gpt-4o-mini \
  --recursive \
  --temperature 0.2 \
  --max-examples 6 \
  --max-example-chars 800
```

---

## 2. Формирование нового артефакта

На этом этапе собираем все подготовленные данные и формируем итоговый артефакт.

### `openapi_to_method_json.py` *(опционально)*  
- Используется, если есть готовое API для функциональности.  
- Позволяет преобразовать OpenAPI в JSON, чтобы LLM могла использовать данные из API при генерации артефакта.  

**Пример запуска:**
```bash
python scripts/openapi_to_method_json.py --spec spec/pro-openapi.yaml --method %Название метода% --minimal -o "source/minimal.json"
```

### `screens_to_requirements.py`  
- Преобразует скрины дизайна в функциональные требования (формат JSON).  

**Пример запуска:**
```bash
python scripts/screens_to_requirements.py ./screens --out "source/requirements.json" --design "source/design.json"
```

### `build_ma_artifact_universal.py`  
- Сборщик итогового промпта.  
- На вход передаются:  
  - сгенерированные функциональные требования,  
  - данные из API (если есть),  
  - профиль стиля,  
  - дополнительные подсказки из файла `hints.json`.  


Перед тем как запускать **build_ma_artifact_universal.py**, рекомендуется выполнить следующие действия:

1. **Файл `style_profile.json`**  
   - Отредактировать перечень желаемых подразделов требований в поле **sections**.  

2. **Файл `minimal.json`**  
   - Проверить и убрать те поля, которые точно не будут использоваться в функциональности.  

3. **Файл `hints.json`**  
   - Передать дополнительный контекст (например, правила сортировки, валидации или особенности отображения).  

**Пример запуска:**
```bash
  python scripts/build_ma_artifact_universal.py \
    --requirements source/requirements.json \
    --api-minimal source/minimal.json \
    --style-profile style/style_profile.json \
    --hints source/hints.json \
    --out "%Название итогового артефакта%" \
    --model gpt-4o
```
---

## 3. Аналитическая работа
 **Финальная проверка**  
   - Проверить результат генерации.  
   - Внести необходимые правки.  
   - Перенести финальную версию артефакта в конфигурацию.  