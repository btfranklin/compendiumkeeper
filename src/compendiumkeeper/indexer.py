# compendiumkeeper/indexer.py
import pickle
from compendiumkeeper.utils import get_embedding_data
from compendiumkeeper.vector_db.pinecone_db import PineconeDB


def index_compendium(compendium_file: str, vector_db_type: str, index_name: str):
    # Load the compendium
    try:
        with open(compendium_file, "rb") as f:
            domain = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading compendium file '{compendium_file}': {e}")

    # Initialize the vector database client
    if vector_db_type == "pinecone":
        vector_db = PineconeDB(index_name=index_name)
    else:
        raise RuntimeError(f"Unsupported vector DB: {vector_db_type}")

    total_concepts = 0
    for topic in domain.topics:
        for concept in topic.concepts:
            embedding_data = get_embedding_data(
                concept, topic.topic_summary, topic.name
            )
            vector_db.upsert_concept_embeddings(embedding_data)
            total_concepts += 1

    print(
        f"Indexed {total_concepts} concepts from domain '{domain.name}' into index '{index_name}'."
    )
