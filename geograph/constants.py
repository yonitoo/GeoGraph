import os
from pathlib import Path
from rdflib import Namespace
from dotenv import load_dotenv

try:
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass 

BGGPT_API_KEY = os.getenv('BGGPT_API_KEY', '')
BGGPT_MODEL = os.getenv('BGGPT_MODEL', 'insait/insait-gemma-2-27b-bg-mixed-19847')  # in the past: "INSAIT-Institute/bggpt-stage3-RFLC-Bigbalance-Duolingual-v2"
BGGPT_V2_API_URL = os.getenv('BGGPT_V2_API_URL', 'https://api.bggpt.ai/completions')
BGGPT_V3_MODEL = os.getenv('BGGPT_V3_MODEL', 'bggpt-gemma-3-27b')
BGGPT_V3_API_URL = os.getenv('BGGPT_V3_API_URL', 'https://api.bggpt.ai/v1/chat/completions/bggpt-gemma-3-27b')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
REASONING_MODEL = "o4-mini"
EMBEDDING_MODEL_NAME = "rmihaylov/roberta-base-nli-stsb-bg"
TOKENIZER_NAME = "rmihaylov/roberta-base-nli-stsb-bg"
TTL_FILE_PATH = str(Path(__file__).parent / "ontologies" / "huge_ontology_v3.ttl")
GRAPHDB_REPO_NAME = 'GeoGraphDB'
TOP_K = 10
EXPANSION_COARSE_TOP_K = 300
EXPANSION_RERANK_TOP_K = 100
EXPANSION_PER_SEED_LIMIT = 100
EXPANSION_BEAM_WIDTH = 50
EXPANSION_FRONTIER_PER_SEED = 20
CHROMA_COLLECTION_NAME = 'bg_geo_huge'
GEOGRAPHBG = Namespace("http://example.org/geographbg#")
ONT_QUANTITY_PROPERTIES = ["надморскаВисочина", "дължинаВБългария", "общаДължина", "площНаВодосборенБасейн", "имаПлощ", "имаНаселение",
                           "имаВисочинаНаПада", "имаОбем", "имаЗалятаПлощ", "имаСреднаНадморскаВисочина", "имаМаксималнаДълбочина",
                           "имаДължинаНаСтена", "имаВисочинаНаСтена", "имаКапацитет", "имаДебит", "имаТемпература", "имаЕстественПрираст", "имаКоефициентНаРаждаемост", "основанПрезГодина", "имаДялОтСветовниЗапаси"]
ONT_VALUE_PROPERTY = "имаСтойност"
ONT_UNIT_PROPERTY = "имаМернаЕдиница"
TYPE_PREFIX = "е тип"
MISSING_VALUE_STRING = "[липсва стойност]"
BASE_EXCLUDED_TYPE_NAMES = {"Мерна Единица", "Количествена Стойност", "Клас", "Property", "Местоположение", "Воден Обект", "Релефна Форма", "Местност"}
UNWANTED_TRIPLE_ENDINGS = {f"{TYPE_PREFIX} {type_name}" for type_name in BASE_EXCLUDED_TYPE_NAMES}
GRAPHDB_REPO_ENDPOINT = 'http://localhost:7200/repositories/GeoGraphDB'

UNWANTED_SECTIONS = ["Вижте също", "Галерия", "Източници", "Външни препратки", "Литература", "Бележки"]
CHROMA_VECTOR_RAG_COLLECTION_NAME = "wiki_bg_geo_chunks"
CHROMA_FIXED256_COLLECTION = "wiki_fixed256"
CHROMA_STRUCTURE_COLLECTION = "wiki_structure"
CHROMA_PARENTCHILD_COLLECTION = "wiki_parentchild"
CHROMA_DB_PATH = "./data/persisted_db_data"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
INGEST_BATCH_SIZE = 500
WIKI_CATEGORY_PAGES_PATH = "./data/deduplicated_data"
PARENT_STORE_PATH = "data/parent_store_parentchild.jsonl"

# Cross-encoder reranker (BAAI/bge-reranker-v2-m3, multilingual, ~2.3 GB download on first use)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Retrieval params for chunking experiment
TOP_N_CANDIDATES = 30        # over-fetch before reranking
CONTEXT_TOKEN_BUDGET = 3000  # max context tokens per query (equalises across strategies)
TOP_K_RERANKED = 5           # final chunks/parents kept after reranking

OPENAI_REASONING_MODEL_SYSTEM_PROMPT = "Ти си AI асистент, който подбира най-полезните RDF тройки за отговор на даден въпрос. Върни резултата САМО и ЕДИНСТВЕНО като валиден Python списък от низове, без никакъв друг текст."

# Minimum number of statements the reasoning filter should keep; if the model
# returns fewer (but non-zero), the pipeline tops up from the reranked list.
REASONING_MIN_KEEP = 10

REASONING_MODEL_PROMPT = f"""
Разполагаш със следния списък от твърдения (факти от граф знания):
--- Начало на Списъка ---
{{kg_triples}}
--- Край на Списъка ---

Въпросът на потребителя е от затворен тип (с варианти за отговор A/B/C/D, изброени във въпроса). Твоята задача е да ПОДБЕРЕШ най-полезните твърдения от горния списък, които биха помогнали на друг модел да избере верния вариант. Ти НЕ отговаряш на въпроса — финалната преценка прави следващият модел; твоята работа е да му дадеш най-добрия наличен материал.
Полезно твърдение е такова, което:
1. Споменава директно ключов обект от въпроса ИЛИ от някой от вариантите за отговор.
2. Описва свойство или характеристика на ключов обект от въпроса или на вариант за отговор (напр. височина, местоположение, тип).
3. Показва връзка между ключови обекти от въпроса или вариантите.
4. Помага да се ИЗКЛЮЧИ (отхвърли) даден вариант — например при въпроси от типа „кое НЕ е вярно“, „кое е грешно/несъответстващо“, или при въпроси за двойки/съответствия, твърдение за САМО ЕДИН от вариантите пак е полезно, защото потвърждава или опровергава именно него.

Подреди избраните твърдения от най-полезно към най-малко полезно.
ВИНАГИ връщай ПОНЕ {REASONING_MIN_KEEP} твърдения (или всички налични, ако списъкът съдържа по-малко). Ако директно релевантните са по-малко от {REASONING_MIN_KEEP}, ДОПЪЛНИ ги с твърденията, които са най-близки по тема до въпроса или до някой от вариантите за отговор, дори да са само косвено свързани. По-добре е да включиш косвено свързано твърдение, отколкото да върнеш твърде кратък списък.

ВАЖНО 1: Ако въпросът се отнася за повече от 1 обект (напр. най-добри 10 неща, или сравнение между вариантите A/B/C/D), подсигури, че след подбора броят тройки е поне толкова, колкото се очаква от въпроса — по възможност поне по едно твърдение за ВСЕКИ вариант за отговор, за да може да се направи сравнение или изключване между тях.

ВАЖНО 2: Върни резултата САМО и ЕДИНСТВЕНО като валиден Python списък от низове (strings), съдържащ подбраните твърдения. Не добавяй никакъв друг текст, обяснения, уводи или заключения преди или след списъка.

Пример за изходен формат:
["Твърдение 1", "Твърдение 5", "Твърдение 12"]

Празен списък [] върни ЕДИНСТВЕНО ако въпросът по принцип не може да бъде отговорен с текст — например ако се отнася до изображение, графика или диаграма, която не е налична. Във всички останали случаи върни поне {REASONING_MIN_KEEP} твърдения.

Въпрос на потребителя:
--- Начало на Въпроса ---
{{user_query}}
--- Край на Въпроса ---
"""

CACHED_HARDCODED_QUERY_PROMPT="""
Използвай САМО предоставения контекст, за да отговориш на въпроса. Ако отговорът не се съдържа ИЗЦЯЛО в контекста, задължително отговори, че не можеш да намериш информацията в предоставените данни - не налучквай.

Контекст:
---------------------
1. Мусала надморскаВисочина 2925 m
2. Вихрен надморскаВисочина 2914 m
3. Кутело надморскаВисочина 2908 m
4. Малка Мусала надморскаВисочина 2902 m
5. Бански суходол надморскаВисочина 2884 m
6. Иречек надморскаВисочина 2852 m
7. Полежан надморскаВисочина 2851 m
8. Малък Полежан надморскаВисочина 2822 m
9. Каменица надморскаВисочина 2822 m
10. Баюви дупки надморскаВисочина 2820 m
---------------------

Въпрос:
---------------------
Кои са десетте най-високи върхове в България?
---------------------

Отговор:
"""

"""
GraphRAG response:
 Десетте най-високи върхове в България, според предоставената информация, са:

1. Мусала - 2925 м
2. Вихрен - 2914 м
3. Кутело - 2908 м
4. Малка Мусала - 2902 м
5. Бански суходол - 2884 м
6. Иречек - 2852 м
7. Полежан - 2851 м
8. Каменица - 2822 м
9. Малък Полежан - 2822 м
10. Баюви дупки - 2820 м

Zero-shot BgGPT:
 Ето списък на десетте най-високи върха в България:

1. Мусала - 2925 м (най-високият връх в Рила и Балканския полуостров)
2. Вихрен - 2914 m (вторият по височина връх в Пирин)
3. Кутело - 2908 m (третият по височина връх в Пирин)
4. Малка Мусала - 2902 m (връх в Рила, близо до Мусала)
5. Иречек - 2822 m (връх в Пирин)
6. Вълчата стена - 2822 m (връх в Стара планина)
7. Голям Полежан - 2822 m (връх в Пирин)
8. Каменица - 2822 m (връх в Пирин)
9. Сиврикая - 2822 m (връх в Пирин)
10. Яловарника - 2765 m (връх в Пирин)
"""
