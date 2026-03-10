# Aiqyn — Development Roadmap

> Offline desktop утилита детекции AI-текста на русском языке.
> Полностью локальная обработка, Windows 10/11 + Linux.

---

## Финальный технологический стек

```
Python 3.12 + uv
├── PySide6 6.7+          — GUI (desktop, zero IPC overhead)
├── pydantic v2           — schemas, config, validation
├── pydantic-settings     — конфиг из TOML + env
├── structlog             — структурированное логирование
├── razdel                — русская токенизация/сегментация
├── spaCy + ru_core_news  — NLP, синтаксический разбор
├── llama-cpp-python      — LLM inference (perplexity, token rank)
├── numpy / scipy         — статистические признаки
├── reportlab             — PDF экспорт (хорошо пакуется)
├── aiosqlite             — история анализов (SQLite async)
├── typer                 — CLI для MVP
├── sentence-transformers — coherence (Фаза 2)
├── lightgbm / catboost   — ML scoring (Фаза 3)
└── pytest + ruff + mypy  — dev tools
```

---

## Структура проекта (целевая)

```
aiqyn/
├── pyproject.toml
├── uv.lock
├── .python-version          # 3.12
├── .gitignore
├── CLAUDE.md
├── ROADMAP.md
├── config/
│   └── default.toml         # дефолтные веса, пороги, модели
├── data/
│   ├── ai_phrases_ru.json   # словарь маркерных фраз ИИ
│   ├── stopwords_ru.txt
│   └── ngram_tables/        # предвычисленные N-gram таблицы
├── models/                  # GGUF файлы (gitignored)
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── human_texts/     # образцы человеческих текстов
│   │   └── ai_texts/        # образцы AI-текстов
│   ├── unit/
│   │   └── extractors/      # тест на каждый экстрактор
│   └── integration/
│       └── test_pipeline.py
└── src/
    └── aiqyn/
        ├── __init__.py
        ├── __main__.py          # точка входа
        ├── config.py            # pydantic-settings + TOML
        ├── logging.py           # structlog setup
        ├── schemas.py           # все Pydantic v2 DTO
        ├── core/
        │   ├── analyzer.py      # TextAnalyzer — главный оркестратор
        │   ├── pipeline.py      # AnalysisPipeline (2-фазовый запуск)
        │   ├── preprocessor.py  # razdel + spaCy preprocessing
        │   ├── aggregator.py    # WeightedSumAggregator + калибровка
        │   └── segmenter.py     # разбивка на сегменты (sliding window)
        ├── extractors/
        │   ├── __init__.py
        │   ├── base.py          # FeatureExtractor Protocol + ExtractionContext
        │   ├── registry.py      # auto-discovery через pkgutil
        │   ├── f01_perplexity.py
        │   ├── f02_burstiness.py
        │   ├── f03_token_entropy.py
        │   ├── f04_lexical_diversity.py
        │   ├── f05_ngram_frequency.py
        │   ├── f06_parse_tree_depth.py
        │   ├── f07_sentence_length.py
        │   ├── f08_punctuation_patterns.py
        │   ├── f09_paragraph_structure.py
        │   ├── f10_ai_phrases.py
        │   ├── f11_emotional_neutrality.py
        │   ├── f12_coherence_smoothness.py
        │   ├── f13_weak_specificity.py
        │   ├── f14_token_rank.py
        │   └── f15_style_consistency.py
        ├── models/
        │   ├── manager.py       # ModelManager singleton
        │   ├── downloader.py    # скачивание GGUF с прогрессом
        │   └── llama_runner.py  # llama-cpp-python wrapper
        ├── storage/
        │   ├── database.py      # aiosqlite, миграции
        │   └── repository.py    # HistoryRepository
        ├── reports/
        │   ├── pdf_exporter.py  # ReportLab PDF
        │   ├── json_exporter.py
        │   └── md_exporter.py
        ├── cli/
        │   └── main.py          # Typer CLI commands
        └── ui/
            ├── app.py           # QApplication + MainWindow
            ├── theme.py         # design tokens, apply_theme()
            ├── views/
            │   ├── main_view.py
            │   ├── result_view.py
            │   ├── history_view.py
            │   └── settings_view.py
            ├── widgets/
            │   ├── drop_zone.py
            │   ├── heatmap_text.py   # HeatmapTextEdit
            │   ├── score_gauge.py    # ScoreGauge (QPainter)
            │   ├── feature_table.py  # QTableView + delegate
            │   ├── segment_sidebar.py
            │   └── progress_panel.py
            └── workers/
                └── analysis_worker.py  # QObject + moveToThread()
```

---

## Ключевые контракты (schemas.py)

```python
from pydantic import BaseModel, Field
from typing import Literal
from enum import StrEnum

class FeatureStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"

class FeatureResult(BaseModel):
    feature_id: str           # "F-01"
    name: str
    value: float | None
    normalized: float | None  # 0.0–1.0
    weight: float
    contribution: float
    status: FeatureStatus = FeatureStatus.OK
    interpretation: str = ""
    error: str | None = None

class SegmentResult(BaseModel):
    id: int
    text: str
    score: float
    label: Literal["human", "ai_generated", "mixed", "unknown"]
    evidence: list[str] = []

class AnalysisMetadata(BaseModel):
    text_length: int
    word_count: int
    language: str
    analysis_time_ms: int
    model_used: str | None
    version: str

class AnalysisResult(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    verdict: str
    confidence: Literal["low", "medium", "high"]
    segments: list[SegmentResult] = []
    features: list[FeatureResult] = []
    metadata: AnalysisMetadata
```

---

## FeatureExtractor Protocol (extractors/base.py)

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import spacy

class FeatureCategory(StrEnum):
    STATISTICAL = "statistical"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    MODEL_BASED = "model_based"
    META = "meta"

@dataclass(frozen=True)
class ExtractionContext:
    raw_text: str
    tokens: list[str]               # razdel токены
    sentences: list[str]            # razdel предложения
    spacy_doc: spacy.tokens.Doc | None = None
    llm: "LLMInference | None" = None

@runtime_checkable
class FeatureExtractor(Protocol):
    @property
    def feature_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def category(self) -> FeatureCategory: ...
    @property
    def requires_llm(self) -> bool: ...
    @property
    def weight(self) -> float: ...
    def extract(self, ctx: ExtractionContext) -> FeatureResult: ...
```

---

## AnalysisPipeline — двухфазовый запуск

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class AnalysisPipeline:
    def run(self, ctx: ExtractionContext) -> list[FeatureResult]:
        extractors = self.registry.get_enabled(self.config)
        non_llm = [e for e in extractors if not e.requires_llm]
        llm_deps = [e for e in extractors if e.requires_llm]

        results = []
        # Фаза 1: параллельно (non-LLM)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._safe_extract, e, ctx): e
                       for e in non_llm}
            for future in as_completed(futures):
                results.append(future.result())

        # Фаза 2: последовательно (один Llama instance)
        for extractor in llm_deps:
            results.append(self._safe_extract(extractor, ctx))

        return results

    def _safe_extract(self, extractor, ctx) -> FeatureResult:
        try:
            return extractor.extract(ctx)
        except Exception as exc:
            return FeatureResult(
                feature_id=extractor.feature_id,
                name=extractor.name,
                value=None, normalized=None,
                weight=0.0, contribution=0.0,
                status=FeatureStatus.FAILED,
                error=str(exc),
            )
```

---

## Фаза 0 — Scaffolding (День 1, ~4 часа)

### Задачи

- [ ] `uv init aiqyn --python 3.12` — создать проект
- [ ] Написать `pyproject.toml` с зависимостями (dev + all extras)
- [ ] `uv sync --extra dev` — проверить
- [ ] `.gitignore` (models/, .venv/, __pycache__, *.gguf, *.db)
- [ ] `config/default.toml` — дефолтные настройки
- [ ] `src/aiqyn/__init__.py` + `__main__.py` — заглушка
- [ ] `src/aiqyn/logging.py` — structlog setup
- [ ] `src/aiqyn/schemas.py` — все DTO
- [ ] `src/aiqyn/extractors/base.py` — Protocol + ExtractionContext
- [ ] `src/aiqyn/extractors/registry.py` — auto-discovery
- [ ] `tests/conftest.py` — базовые фикстуры
- [ ] Скачать spaCy модель: `uv run python -m spacy download ru_core_news_sm`
- [ ] Скачать модель: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (~4.7 GB) в `models/`

### pyproject.toml

```toml
[project]
name = "aiqyn"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "structlog>=24.1",
    "razdel>=0.5",
    "typer>=0.12",
    "tomli>=2.0; python_version < '3.11'",
]

[project.optional-dependencies]
nlp = ["spacy>=3.7"]
llm = ["llama-cpp-python>=0.2.80"]
ui  = ["PySide6>=6.7", "reportlab>=4.2"]
db  = ["aiosqlite>=0.20"]
ml  = ["lightgbm>=4.3", "scipy>=1.13", "numpy>=1.26"]
dev = ["pytest>=8.2", "pytest-asyncio>=0.23", "ruff>=0.4", "mypy>=1.10"]
all = ["aiqyn[nlp,llm,ui,db,ml]"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.12"
strict = true
```

---

## Фаза 1 — MVP CLI (Недели 1–3)

**Цель:** `uv run aiqyn analyze text.txt` → JSON с overall_score + 6 признаков

### Sprint 1.1 — Core contracts (Дни 1–3)

- [ ] `schemas.py` — все Pydantic v2 DTO (FeatureResult, AnalysisResult, SegmentResult...)
- [ ] `extractors/base.py` — FeatureExtractor Protocol + ExtractionContext
- [ ] `extractors/registry.py` — pkgutil auto-discovery
- [ ] `config.py` — pydantic-settings + TOML
- [ ] `core/preprocessor.py` — razdel tokenize/sentenize + spaCy lazy init
- [ ] Unit-тест: preprocessor возвращает корректный ExtractionContext

### Sprint 1.2 — 6 MVP Extractors (Дни 4–10)

Порядок реализации (от простого к сложному):

| Файл | Feature | Зависимости | Сложность |
|---|---|---|---|
| `f02_burstiness.py` | std длин предложений | numpy | Низкая |
| `f04_lexical_diversity.py` | TTR + hapax legomena | — | Низкая |
| `f07_sentence_length.py` | mean/std/percentiles | numpy | Низкая |
| `f10_ai_phrases.py` | словарь маркеров → regex | json | Низкая |
| `f11_emotional_neutrality.py` | тональный профиль | spaCy | Средняя |
| `f01_perplexity.py` | log-prob через LLM | llama-cpp | Высокая |

Для каждого экстрактора: **реализация + unit-тест с fixture-текстом**.

### Sprint 1.3 — Pipeline + Aggregator + CLI (Дни 11–15)

- [ ] `core/pipeline.py` — двухфазовый запуск (ThreadPoolExecutor + sequential LLM)
- [ ] `core/aggregator.py` — взвешенная сумма + `confidence` по spread признаков
- [ ] `models/manager.py` — ModelManager singleton (один Llama instance)
- [ ] `models/llama_runner.py` — wrapper с retry + timeout
- [ ] `cli/main.py` — Typer CLI:
  - `aiqyn analyze <file>` — анализ файла
  - `aiqyn analyze -` — текст из stdin
  - `aiqyn analyze <file> --format json --output result.json`
  - `aiqyn analyze <file> --no-llm` — только rule-based
- [ ] Integration test: полный pipeline на 3 fixture-текстах (human / ai / mixed)

**Deliverables Фазы 1:**
- CLI работает end-to-end
- Graceful degradation при недоступной LLM (`--no-llm`)
- JSON вывод валидируется Pydantic схемой
- Coverage ≥ 80% для extractors

---

## Фаза 2 — PySide6 GUI + все признаки (Недели 4–8)

**Цель:** полноценный desktop, все 15 экстракторов, heatmap, экспорт PDF

### Sprint 2.1 — PySide6 scaffold + MVP UI (Дни 16–22)

- [ ] `ui/app.py` — QApplication + MainWindow + QStackedWidget
- [ ] `ui/theme.py` — design tokens (тёмная тема по умолчанию), `apply_theme()`
- [ ] `ui/views/main_view.py` — InputArea + DropZone + StartButton
- [ ] `ui/widgets/drop_zone.py` — dragEnterEvent / dropEvent (txt, docx, pdf)
- [ ] `ui/widgets/progress_panel.py` — многоэтапный прогресс с названиями этапов
- [ ] `ui/workers/analysis_worker.py` — QObject + moveToThread():
  - Signals: `progress(stage_id, pct)`, `segment_ready(dict)`, `finished(dict)`, `error(str)`
  - `cancel()` через threading.Event флаг
- [ ] Кнопка «Отмена» работает корректно

### Sprint 2.2 — ResultView + ключевые виджеты (Дни 23–29)

- [ ] `ui/widgets/heatmap_text.py` — HeatmapTextEdit:
  - QTextEdit readonly + QTextCharFormat раскраска по score
  - Клик на абзац → `segment_selected(segment_id)` signal
  - Live-update: абзацы раскрашиваются по мере готовности сегментов
- [ ] `ui/widgets/score_gauge.py` — ScoreGauge:
  - Кастомный QPainter (полукруг/дуга)
  - QPropertyAnimation при появлении результата
  - Цвет: зелёный → жёлтый → красный
- [ ] `ui/widgets/feature_table.py` — QTableView + QAbstractTableModel:
  - Столбцы: признак / значение / вклад / интерпретация
  - ContributionDelegate: mini progress bar в ячейке вклада
- [ ] `ui/widgets/segment_sidebar.py` — детали выбранного сегмента
- [ ] `ui/views/result_view.py` — компоновка: HeatmapText + ScoreGauge + FeatureTable + Sidebar

### Sprint 2.3 — Оставшиеся 9 экстракторов (параллельно с 2.2)

| Файл | Feature | Зависимости |
|---|---|---|
| `f03_token_entropy.py` | энтропия токенов | numpy |
| `f05_ngram_frequency.py` | частотность N-gram | data/ngram_tables/ |
| `f06_parse_tree_depth.py` | глубина синтаксических деревьев | spaCy dep parser |
| `f08_punctuation_patterns.py` | паттерны пунктуации | regex |
| `f09_paragraph_structure.py` | тезис→аргументы→вывод | spaCy |
| `f12_coherence_smoothness.py` | косинусное сходство предложений | sentence-transformers |
| `f13_weak_specificity.py` | плотность NER-сущностей | spaCy NER |
| `f14_token_rank.py` | rank топ-K токенов | llama-cpp logits |
| `f15_style_consistency.py` | дрейф стиля между сегментами | метрики из других F |

**Оптимизация LLM:** `f01_perplexity.py` + `f14_token_rank.py` — один forward pass через `LLMAnalysisBundle`.

### Sprint 2.4 — Сегментный анализ + История + Экспорт (Дни 30–35)

- [ ] `core/segmenter.py` — sliding window (3–5 предложений, min 50 слов/сегмент)
- [ ] `storage/database.py` — aiosqlite, DDL, миграции через версионирование
- [ ] `storage/repository.py` — HistoryRepository (save / list / get / delete)
- [ ] `ui/views/history_view.py` — QListView + QAbstractListModel + поиск
- [ ] `reports/pdf_exporter.py` — ReportLab: вердикт + таблица признаков + сегменты
- [ ] `reports/json_exporter.py` + `reports/md_exporter.py`
- [ ] `ui/views/result_view.py` — кнопки «Сохранить PDF» / «Экспорт JSON»

**Deliverables Фазы 2:**
- Полный desktop с GUI
- Все 15 экстракторов (graceful degradation при ошибке любого)
- Посегментный анализ + heatmap с live-update
- История в SQLite
- PDF / JSON / Markdown экспорт
- Тёмная тема по умолчанию, переключение в Settings

---

## Фаза 3 — Production-ready + Installer (Недели 9–12)

**Цель:** .exe инсталлятор, калибровка, model manager UI

### Sprint 3.1 — Model Manager + Calibration (Дни 36–45)

- [ ] `models/downloader.py` — скачивание GGUF при первом запуске:
  - Список моделей из `data/models_catalog.json`
  - Download в `%APPDATA%\Aiqyn\models\` (Windows) / `~/.local/share/aiqyn/models/` (Linux)
  - QThread с прогрессом + resume при обрыве
- [ ] `ui/views/settings_view.py` — ModelManagerDialog, пороги чувствительности, тема
- [ ] Сбор датасета: 300+ текстов (human / ai-generated) для калибровки
- [ ] `core/calibrator.py` — Platt scaling (LogisticRegression из scikit-learn)
- [ ] Benchmark: измерение F1, precision, recall, AUC на тестовом датасете

### Sprint 3.2 — Packaging (Дни 46–50)

**PyInstaller конфиг (`aiqyn.spec`):**
```python
# hook-llama_cpp.py — кастомный хук
from PyInstaller.utils.hooks import collect_dynamic_libs
binaries = collect_dynamic_libs('llama_cpp')
```

```bash
# CPU-only сборка
pyinstaller --onedir \
  --name "Aiqyn" \
  --add-data "src/aiqyn/data:aiqyn/data" \
  --add-data "config:config" \
  --hidden-import spacy \
  --collect-all spacy \
  src/aiqyn/__main__.py
```

**Inno Setup (`installer.iss`):**
- `PrivilegesRequired=lowest` — установка без UAC
- Директория: `{localappdata}\Programs\Aiqyn`
- Модели: `{userappdata}\Aiqyn\models\` (не трогать при обновлении!)
- Создать shortcut в Start Menu + Desktop (опционально)

**GitHub Actions CI (`build.yml`):**
```yaml
jobs:
  build:
    strategy:
      matrix:
        variant: [cpu, cuda]
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra all
      - run: uv run pytest tests/ -x
      - run: uv run pyinstaller aiqyn.spec
      - run: iscc installer.iss
      - uses: actions/upload-artifact@v4
        with:
          name: Aiqyn-${{ matrix.variant }}-installer
          path: Output/*.exe
```

**Deliverables Фазы 3:**
- `Aiqyn-cpu-setup.exe` < 500 MB (без модели)
- `Aiqyn-cuda-setup.exe` — с CUDA поддержкой
- Установка без UAC на корпоративных машинах
- Smoke test в CI на каждый PR
- Model manager скачивает модель при первом запуске с прогрессом

---

## Docker — стратегия контейнеризации

Десктопный GUI в Docker не запускается — PySide6 требует дисплей.
Docker используется в **трёх сценариях**:

### Сценарий 1 — Dev Container (`.devcontainer/`)

Единая воспроизводимая среда для разработчиков: Python 3.12, uv, spaCy, llama-cpp-python, все dev-зависимости. Без необходимости ставить что-то локально.

```
.devcontainer/
├── devcontainer.json
└── Dockerfile
```

**`.devcontainer/Dockerfile`:**
```dockerfile
FROM python:3.12-slim-bookworm

# System deps для spaCy + llama-cpp + сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra dev --extra nlp --extra db --extra ml

# spaCy модель
RUN uv run python -m spacy download ru_core_news_sm

ENV PATH="/workspace/.venv/bin:$PATH"
```

**`.devcontainer/devcontainer.json`:**
```json
{
  "name": "Aiqyn Dev",
  "build": { "dockerfile": "Dockerfile" },
  "mounts": [
    "source=${localWorkspaceFolder}/models,target=/workspace/models,type=bind"
  ],
  "features": {
    "ghcr.io/devcontainers/features/git:1": {}
  },
  "postCreateCommand": "uv sync --extra dev",
  "remoteUser": "root"
}
```

> Модели монтируются из хоста — не хранятся в образе.

---

### Сценарий 2 — Inference Service (API mode, v1.5+)

Headless backend для API-режима: FastAPI сервер + llama-cpp + все extractors.
GUI при этом подключается к нему локально (или по сети в enterprise-сценарии).

```
docker/
├── Dockerfile.cpu          # CPU-only inference
├── Dockerfile.cuda         # NVIDIA GPU inference
└── docker-compose.yml      # dev/prod оркестрация
```

**`docker/Dockerfile.cpu`:**
```dockerfile
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra nlp --extra llm --extra db --no-dev

COPY src/ ./src/
RUN uv run python -m spacy download ru_core_news_sm

# ---- runtime ----
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY config/ ./config/
COPY data/ ./data/

ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

EXPOSE 8080
CMD ["python", "-m", "aiqyn", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

**`docker/Dockerfile.cuda`:**
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip \
    build-essential cmake cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./

# llama-cpp-python с CUDA
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1
RUN uv sync --extra nlp --extra llm --extra db --no-dev

COPY src/ ./src/
RUN uv run python -m spacy download ru_core_news_sm

# ---- runtime ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY config/ ./config/
COPY data/ ./data/

ENV PATH="/app/.venv/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD python3.12 -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

EXPOSE 8080
CMD ["python3.12", "-m", "aiqyn", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

**`docker/docker-compose.yml`:**
```yaml
services:
  aiqyn-cpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cpu
    image: aiqyn:cpu
    container_name: aiqyn-inference-cpu
    ports:
      - "8080:8080"
    volumes:
      - models:/app/models
      - ${APPDATA:-~/.local/share}/Aiqyn/history:/app/data/history
    environment:
      - AIQYN_LOG_LEVEL=INFO
      - AIQYN_MODEL_PATH=/app/models
    restart: unless-stopped

  aiqyn-cuda:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cuda
    image: aiqyn:cuda
    container_name: aiqyn-inference-cuda
    ports:
      - "8080:8080"
    volumes:
      - models:/app/models
      - ${APPDATA:-~/.local/share}/Aiqyn/history:/app/data/history
    environment:
      - AIQYN_LOG_LEVEL=INFO
      - AIQYN_MODEL_PATH=/app/models
      - AIQYN_GPU_LAYERS=35
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    profiles: [cuda]

volumes:
  models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./models
```

**Запуск:**
```bash
# CPU режим
docker compose -f docker/docker-compose.yml up aiqyn-cpu

# CUDA режим
docker compose -f docker/docker-compose.yml --profile cuda up aiqyn-cuda
```

---

### Сценарий 3 — CI Build Container

PyInstaller Windows .exe собирается на `windows-2022` runner (Docker не нужен для .exe).
Docker используется для **Linux-сборки** и **тестов** в CI.

**`.github/workflows/build.yml`:**
```yaml
jobs:
  test-linux:
    runs-on: ubuntu-latest
    container:
      image: python:3.12-slim-bookworm
    steps:
      - uses: actions/checkout@v4
      - run: pip install uv && uv sync --extra dev --extra nlp --extra db
      - run: uv run python -m spacy download ru_core_news_sm
      - run: uv run pytest tests/ -x --tb=short

  build-windows:
    needs: test-linux
    runs-on: windows-2022
    strategy:
      matrix:
        variant: [cpu, cuda]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra all
      - run: uv run pyinstaller aiqyn.spec
      - run: iscc installer.iss
      - uses: actions/upload-artifact@v4
        with:
          name: Aiqyn-${{ matrix.variant }}-setup
          path: Output/*.exe

  build-docker:
    needs: test-linux
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.cpu
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

### Финальная структура файлов Docker

```
aiqyn/
├── .devcontainer/
│   ├── Dockerfile           # dev environment
│   └── devcontainer.json
└── docker/
    ├── Dockerfile.cpu       # inference service (CPU)
    ├── Dockerfile.cuda      # inference service (NVIDIA GPU)
    └── docker-compose.yml   # cpu + cuda profiles
```

---

## Фаза 4 — v1.5+ (После релиза)

- [ ] Batch-обработка: перетащить папку → анализ всех .txt/.docx/.pdf
- [ ] API-режим: `aiqyn serve --port 8080` (FastAPI, локальный)
- [ ] Плагин для Microsoft Word (COM/VSTO или web add-in)
- [ ] Расширение языков: английский текст (отдельный набор признаков)
- [ ] Автообновление приложения (Sparkle / self-update mechanism)
- [ ] LightGBM / CatBoost замена rule-based aggregation

---

## Таблица признаков — приоритет и веса

| ID | Признак | Фаза | Вес MVP | Вес v1.0 | LLM |
|---|---|---|---|---|---|
| F-01 | Perplexity | 1 | 0.25 | 0.18 | Да |
| F-02 | Burstiness | 1 | 0.20 | 0.12 | Нет |
| F-04 | Lexical diversity (TTR) | 1 | 0.15 | 0.10 | Нет |
| F-07 | Sentence length distribution | 1 | 0.15 | 0.08 | Нет |
| F-10 | AI marker phrases (RU) | 1 | 0.15 | 0.10 | Нет |
| F-11 | Emotional neutrality | 1 | 0.10 | 0.08 | Нет |
| F-03 | Token entropy | 2 | — | 0.06 | Нет |
| F-05 | N-gram frequency | 2 | — | 0.06 | Нет |
| F-06 | Parse tree depth | 2 | — | 0.05 | Нет |
| F-08 | Punctuation patterns | 2 | — | 0.04 | Нет |
| F-09 | Paragraph structure | 2 | — | 0.04 | Нет |
| F-12 | Coherence/smoothness | 2 | — | 0.06 | Нет |
| F-13 | Weak specificity (NER) | 2 | — | 0.05 | Нет |
| F-14 | Token rank | 2 | — | 0.10 | Да |
| F-15 | Style consistency | 2 | — | 0.08 | Нет |

---

## Риски и митигация

| Риск | Вероятность | Митигация |
|---|---|---|
| PyInstaller не пакует llama-cpp-python с CUDA | Высокая | Кастомный `hook-llama_cpp.py`; Nuitka как fallback |
| 16 GB RAM мало для 7B + spaCy + GUI | Средняя | Q4_K_S (3.8 GB) вместо Q4_K_M; lazy load spaCy |
| Perplexity > 30 сек на CPU для длинных текстов | Высокая | Max 4096 токенов через LLM; остальное rule-based |
| spaCy некачественно парсит синтаксис | Средняя | `ru_core_news_lg` вместо `sm`; низкий вес F-06 |
| False positive rate > 15% | Средняя | Platt calibration; консервативные пороги в MVP |
| CUDA Runtime отсутствует у пользователя | Средняя | Проверка при установке, fallback на CPU, ссылка на CUDA Toolkit |

---

## Немедленные action items (первые 2 дня)

```bash
# День 1
uv init aiqyn --python 3.12
cd aiqyn
# → создать pyproject.toml (см. выше)
uv sync --extra dev
uv run python -m spacy download ru_core_news_sm
mkdir -p src/aiqyn/{core,extractors,models,storage,reports,cli,ui} tests/{unit/extractors,integration} config data models

# День 2
# → реализовать schemas.py
# → реализовать extractors/base.py
# → реализовать extractors/registry.py
# → написать f02_burstiness.py + тест
# → uv run pytest → green
```
