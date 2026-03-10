# Aiqyn — Детектор ИИ-текста

Офлайн-инструмент для определения AI-сгенерированного текста на **русском языке**.
Все вычисления выполняются локально — никаких облачных сервисов, никакой телеметрии.

---

## Как это работает

Текст прогоняется через **15 независимых признаков** из пяти категорий:

| Категория | Признаки |
|---|---|
| Статистические | Перплексия, Бёрстинес, Энтропия токенов, TTR/hapax, N-gram частотность |
| Синтаксические | Глубина дерева разбора, Длины предложений, Пунктуация, Структура абзацев |
| Семантические | Маркерные фразы ИИ, Эмоциональный нейтралитет, Когерентность, Слабая конкретика |
| Модельные | Перплексия (Ollama), Ранг токенов |
| Мета | Консистентность стиля |

Результат — взвешенная сумма 0–100%, вердикт и тепловая карта по сегментам текста.

---

## Быстрый старт

### Docker (рекомендуется)

```bash
git clone https://github.com/ssb000ss/Aiqyn.git
cd Aiqyn
docker compose up
```

Открыть в браузере: **http://localhost:8000**

Для работы с GPU (требует nvidia-docker):

```bash
docker compose -f docker/docker-compose.yml --profile cuda up aiqyn-cuda
```

### Локально

```bash
# Установить зависимости
uv sync

# Загрузить русскую языковую модель spaCy
uv run python -m spacy download ru_core_news_sm

# Запустить веб-интерфейс
uv run uvicorn aiqyn.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Открыть в браузере: **http://localhost:8000**

### Ollama (опционально, для LLM-признаков)

```bash
# Установить Ollama: https://ollama.com
ollama pull qwen3:8b
```

Без Ollama все 15 признаков продолжают работать — только F-01 и F-14 переходят в fallback-режим.

---

## Интерфейс

Веб-интерфейс (SPA на Alpine.js + Tailwind):

- **Анализ** — вставьте текст или загрузите файл (.txt, .docx, .pdf)
- **Результат** — gauge с процентом, вердикт, аккордион признаков, тепловая карта
- **История** — все предыдущие анализы, поиск, повторное открытие
- **Настройки** — статус системы, переключение LLM-режима

---

## CLI

```bash
# Анализ файла
uv run python -m aiqyn analyze text.txt

# Из stdin
cat text.txt | uv run python -m aiqyn analyze -

# Без LLM (быстрее)
uv run python -m aiqyn analyze text.txt --no-llm

# JSON-вывод
uv run python -m aiqyn analyze text.txt --format json
```

---

## REST API

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/health` | Проверка доступности |
| `GET` | `/status` | Версия, статус Ollama, активная модель |
| `POST` | `/analyze` | Анализ текста |
| `POST` | `/upload` | Извлечение текста из .docx / .pdf |
| `GET` | `/history` | Список последних анализов |
| `GET` | `/history/{id}` | Полный результат по ID |
| `DELETE` | `/history/{id}` | Удалить запись |

Swagger UI доступен по адресу `/docs`.

**Пример запроса:**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Ваш текст здесь...", "use_llm": false}'
```

**Пример ответа:**

```json
{
  "overall_score": 0.74,
  "verdict": "Вероятно сгенерировано ИИ",
  "confidence": "high",
  "features": [...],
  "segments": [...],
  "metadata": {
    "word_count": 312,
    "analysis_time_ms": 4200,
    "model_used": null,
    "language": "ru"
  }
}
```

---

## Структура проекта

```
src/aiqyn/
├── api/            # FastAPI приложение и схемы запросов
├── cli/            # Typer CLI команды
├── core/           # Оркестратор, пайплайн, препроцессор, агрегатор
├── extractors/     # 15 экстракторов признаков (f01–f15)
├── models/         # Обёртки над Ollama и llama-cpp
├── reports/        # Экспорт в PDF, JSON, Markdown
├── storage/        # SQLite история (CRUD)
├── ui/             # PySide6 десктоп (legacy)
├── web/            # Веб-роутер и HTML-шаблон
└── utils/          # Чтение .txt / .docx / .pdf
config/
└── default.toml    # Веса признаков, пороги, параметры модели
data/
├── ai_phrases_ru.json   # ~450 маркерных фраз + 60 regex
└── sentiment_ru.json    # ~820 seed-лемм → 18 000+ словоформ
```

---

## Конфигурация

Настройки читаются из `config/default.toml`, переопределяются переменными окружения:

```toml
[ollama]
base_url = "http://localhost:11434"
model    = "qwen3:8b"

[thresholds]
human        = 0.35   # score < 0.35 → человек
ai_generated = 0.65   # score > 0.65 → ИИ

[analysis]
max_text_length = 50000
```

| Переменная | Описание |
|---|---|
| `AIQYN_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` |
| `AIQYN_USE_LLM` | `true` / `false` |
| `AIQYN_MODEL_PATH` | Путь к GGUF-модели |
| `AIQYN_GPU_LAYERS` | Кол-во слоёв на GPU (0 = CPU) |

---

## Разработка

```bash
# Установить dev-зависимости
uv sync

# Тесты
uv run pytest
uv run pytest tests/unit/extractors/test_burstiness.py  # один тест

# Линтинг
uv run ruff check src/
uv run ruff format src/
```

---

## Технологии

- **Python 3.12+**, uv, ruff
- **FastAPI** + Uvicorn, Pydantic v2, Jinja2
- **spaCy** (`ru_core_news_sm`), razdel, pymorphy3
- **Alpine.js** + Tailwind CSS (SPA без сборки)
- **SQLite** (история анализов)
- **Ollama** / llama-cpp-python (LLM-признаки, опционально)
- **PySide6** (legacy desktop UI)

---

## Ограничения

- Тексты на **русском языке** (другие языки не калиброваны)
- Результаты **вероятностные** — не являются юридическим доказательством
- LLM-признаки (F-01, F-14) требуют запущенного Ollama; без него работают в fallback-режиме
- Минимальный размер текста — **30 слов**

---

## Лицензия

MIT © 2026 ssb000ss
