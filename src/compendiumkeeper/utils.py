import re
import openai
import os


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def generate_concept_id(topic_name: str, concept_name: str) -> str:
    return f"{slugify(topic_name)}_{slugify(concept_name)}"


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return api_key


def get_embedding(text: str) -> list[float]:
    openai.api_key = get_openai_api_key()
    try:
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error generating embedding for text '{text}': {e}")
        return []


def get_embedding_data(concept, topic_summary: str, topic_name: str):
    concept_id = generate_concept_id(topic_name, concept.name)

    name_text = concept.name
    content_text = f"{topic_summary}\n\n{concept.content}"
    question_texts = concept.questions
    keyword_texts = concept.keywords
    combined_keywords = " ".join(keyword_texts) if keyword_texts else None

    embedding_data = {
        "concept_id": concept_id,
        "name": (name_text, get_embedding(name_text)),
        "content": (content_text, get_embedding(content_text)),
        "questions": [(q, get_embedding(q)) for q in question_texts],
        "keywords": [(kw, get_embedding(kw)) for kw in keyword_texts],
        "combined_keywords": (
            (combined_keywords, get_embedding(combined_keywords))
            if combined_keywords
            else None
        ),
    }

    return embedding_data
