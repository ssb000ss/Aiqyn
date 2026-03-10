"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from aiqyn.extractors.base import ExtractionContext
from aiqyn.logging import setup_logging

setup_logging(level="WARNING")


HUMAN_TEXT = """
Сегодня я наконец-то дошёл до той книги, которую откладывал несколько месяцев. 
Знаете, иногда бывает такое: понимаешь, что вещь важная, но руки не доходят. 
А тут раз — и прочитал за выходные запоем. Это была «Мастер и Маргарита». 
Да, классика, и все её читали, но я как-то умудрился обойти стороной. 
Если честно — не ожидал такого. Ощущение странное: смешно и жутко одновременно. 
Воланд там вообще потрясающий персонаж — такой многослойный, не однозначный. 
Думал, будет скучновато, а оказалось — не мог оторваться. Ладно, хватит спойлеров.
""".strip()

AI_TEXT = """
Данная тема является весьма актуальной в современном мире. 
Необходимо отметить, что рассматриваемый вопрос имеет важное значение для общества. 
С одной стороны, существуют определённые преимущества данного подхода. 
С другой стороны, следует учитывать возможные недостатки и ограничения. 
Таким образом, можно констатировать, что проблема требует комплексного рассмотрения. 
В заключение следует подчеркнуть, что данный вопрос нуждается в дальнейшем изучении. 
Подводя итог вышесказанному, можно сделать вывод о необходимости системного подхода.
""".strip()


@pytest.fixture
def human_ctx() -> ExtractionContext:
    from aiqyn.core.preprocessor import TextPreprocessor
    pp = TextPreprocessor(load_spacy=False)
    return pp.process(HUMAN_TEXT)


@pytest.fixture
def ai_ctx() -> ExtractionContext:
    from aiqyn.core.preprocessor import TextPreprocessor
    pp = TextPreprocessor(load_spacy=False)
    return pp.process(AI_TEXT)


@pytest.fixture
def preprocessor():
    from aiqyn.core.preprocessor import TextPreprocessor
    return TextPreprocessor(load_spacy=False)
