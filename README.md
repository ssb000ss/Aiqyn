# Aiqyn — Детектор ИИ-текста

Офлайн-инструмент для определения AI-сгенерированного текста на **русском языке**.
Все вычисления выполняются локально — никаких облачных сервисов, никакой телеметрии.

---

## Как это работает

Текст прогоняется через **17 независимых признаков** из пяти категорий:

| Категория | Признаки |
|---|---|
| Статистические | Перплексия, Бёрстинес, Энтропия токенов, TTR/hapax, N-gram частотность |
| Синтаксические | Глубина дерева разбора, Длины предложений, Пунктуация, Структура абзацев |
| Семантические | Маркерные фразы ИИ, Эмоциональный нейтралитет, Когерентность, Слабая конкретика |
| Модельные | Перплексия (Ollama), Ранг токенов, **Binoculars** (двухмодельный ratio), **RuBERT-tiny2** |
| Мета | Консистентность стиля |

Результат — взвешенная сумма 0–100%, вердикт, тепловая карта по сегментам и список доказательств
(*evidence*). Веса подобраны эмпирически на парных сэмплах; дополнительно доступна
**Platt-калибровка** — при наличии `data/calibration.json` raw-скор корректируется сигмоидой.

> F-16 Binoculars и F-17 RuBERT-tiny2 отключены по умолчанию (`weight = 0.0`) — требуют
> калибровки на реальном датасете. Остальные 15 активны.

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
ollama pull qwen3:8b          # основная модель (F-01, F-14)
ollama pull qwen3:1.7b        # опционально: secondary для F-16 Binoculars
ollama pull nomic-embed-text  # опционально: embedding-путь для F-12 когерентности
```

Без Ollama все 15 активных признаков продолжают работать — F-01 и F-14 переходят в fallback.
Без `nomic-embed-text` F-12 падает на spaCy-леммы (если установлена модель) или surface-токены.

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
# Анализ файла (LLM-признаки включены, если Ollama доступен)
uv run python -m aiqyn analyze text.txt

# Из stdin
cat text.txt | uv run python -m aiqyn analyze -

# Без LLM — в 20-30 раз быстрее, работает полностью офлайн
uv run python -m aiqyn analyze text.txt --no-llm

# JSON-вывод
uv run python -m aiqyn analyze text.txt --format json
```

Пример вывода:

```
  Вердикт:      Вероятно сгенерировано ИИ
  Вероятность:  71.5%  [██████████░░░░░]
  Уверенность:  medium
  Время:        8.46 сек
  Слов:         118
  Признаки:
    f10_ai_phrases           100.0%  Высокая плотность маркерных фраз ИИ
    f02_burstiness            99.7%  Низкая вариативность предложений
    f03_token_entropy         71.2%  Формальная лексика (длина слов=8.3)
    ...
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
├── core/           # Оркестратор, пайплайн, препроцессор, агрегатор, Platt-калибратор
├── extractors/     # 17 экстракторов признаков (f01–f17, f16/f17 off by default)
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
├── sentiment_ru.json    # ~820 seed-лемм → 18 000+ словоформ
└── calibration.json     # (опционально) Platt-коэффициенты A, B
```

---

## Конфигурация

Настройки читаются из `config/default.toml`, переопределяются переменными окружения с префиксом
`AIQYN_`.

```toml
[ollama]
base_url        = "http://localhost:11434"
model           = "qwen3:8b"
secondary_model = "qwen3:1.7b"       # для F-16 Binoculars
embed_model     = "nomic-embed-text" # для F-12 через embeddings

[thresholds]
human        = 0.35   # score < 0.35 → человек
ai_generated = 0.65   # score > 0.65 → ИИ

[analysis]
max_text_length = 50000
```

| Переменная | Описание |
|---|---|
| `AIQYN_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` |
| `AIQYN_TEXT_DOMAIN` | `general` (блоги, соцсети) / `formal` (документы, по умолчанию) — переключает набор весов |
| `AIQYN_OLLAMA_MODEL` | Основная модель для F-01 / F-14 (default: `qwen3:8b`) |
| `AIQYN_OLLAMA_SECONDARY_MODEL` | Вторая модель для F-16 Binoculars (`""` = отключить) |
| `AIQYN_OLLAMA_EMBED_MODEL` | Embedding-модель для F-12 (default: `nomic-embed-text`) |
| `AIQYN_CALIBRATION_PATH` | Путь к `calibration.json` с Platt-коэффициентами; `disabled` = всегда сырой score |
| `AIQYN_SEGMENT_WEIGHT` | `0.0`–`1.0`: подмешивать ли среднее по сегментам в общий скор (default `0.0` — не мешать) |
| `AIQYN_EVIDENCE_TOP_N` | Сколько признаков показывать в списке evidence (default `5`) |
| `AIQYN_THRESHOLD_HUMAN` / `AIQYN_THRESHOLD_AI` | Пороги вердикта (default `0.35` / `0.65`) |
| `AIQYN_MODEL_PATH` | Путь к GGUF-модели (fallback, если Ollama недоступен) |
| `AIQYN_GPU_LAYERS` | Кол-во слоёв на GPU (0 = CPU) |

---

## Разработка

```bash
# Установить dev-зависимости
uv sync --extra dev

# Тесты (~287 unit + интеграционные, прогон < 2 сек)
uv run pytest
uv run pytest tests/unit/extractors/test_f16_binoculars.py -v  # один модуль

# Линтинг
uv run ruff check src/
uv run ruff format src/
```

Опциональные группы зависимостей:

| Extra | Для чего |
|---|---|
| `dev` | pytest, pytest-asyncio, ruff, mypy |
| `nlp` | spaCy + sentence-transformers |
| `hf` | transformers + torch (нужен для F-17 RuBERT) |
| `llm` | llama-cpp-python (CPU/GPU fallback, если нет Ollama) |
| `ui` | PySide6 + reportlab (legacy desktop) |

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
- F-12 когерентность показывает максимальное качество только с `nomic-embed-text` или
  установленной моделью `ru_core_news_sm` для spaCy — без них падает на surface-токены
- F-16 / F-17 отключены по умолчанию (`weight = 0.0`) — требуют калибровки на датасете
- Минимальный размер текста — **30 слов**

---

## Лицензия

MIT © 2026 ssb000ss
