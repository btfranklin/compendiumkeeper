import os
from pinecone import Pinecone, ServerlessSpec
from compendiumkeeper.vector_db.base import VectorDatabase


class PineconeDB(VectorDatabase):
    def __init__(self, index_name: str):
        """
        Initialize PineconeDB with the specified index configurations.

        Args:
            index_name (str): Name of the Pinecone index.
        """
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY must be set in the .env file.")

        # Initialize the Pinecone client
        self.pc = Pinecone(api_key=api_key)

        # Hard-coded defaults for this application
        dimension = 1536  # Suitable for text-embedding-ada-002
        metric = "cosine"
        spec = ServerlessSpec(
            cloud="aws", region="us-east-1"
        )  # Adjust region as needed

        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes():
            try:
                self.pc.create_index(
                    name=index_name, dimension=dimension, metric=metric, spec=spec
                )
                print(
                    f"Pinecone index '{index_name}' created with dimension {dimension}, metric '{metric}', cloud '{spec.cloud}', region '{spec.region}'."
                )
            except Exception as e:
                raise RuntimeError(f"Error creating Pinecone index '{index_name}': {e}")
        else:
            print(f"Pinecone index '{index_name}' already exists.")

        self.index = self.pc.Index(index_name)

    def upsert_concept_embeddings(self, embedding_data: dict):
        vectors = []
        concept_id = embedding_data["concept_id"]

        # Name
        name_text, name_embedding = embedding_data["name"]
        vectors.append(
            (
                f"{concept_id}_name",
                name_embedding,
                {"type": "name", "text": name_text, "concept_id": concept_id},
            )
        )

        # Content
        content_text, content_embedding = embedding_data["content"]
        vectors.append(
            (
                f"{concept_id}_content",
                content_embedding,
                {"type": "content", "text": content_text, "concept_id": concept_id},
            )
        )

        # Questions
        for i, (q_text, q_emb) in enumerate(embedding_data["questions"]):
            vectors.append(
                (
                    f"{concept_id}_question_{i}",
                    q_emb,
                    {"type": "question", "text": q_text, "concept_id": concept_id},
                )
            )

        # Keywords
        for i, (kw_text, kw_emb) in enumerate(embedding_data["keywords"]):
            vectors.append(
                (
                    f"{concept_id}_keyword_{i}",
                    kw_emb,
                    {"type": "keyword", "text": kw_text, "concept_id": concept_id},
                )
            )

        # Combined Keywords
        if embedding_data["combined_keywords"] is not None:
            ck_text, ck_emb = embedding_data["combined_keywords"]
            vectors.append(
                (
                    f"{concept_id}_combined_keywords",
                    ck_emb,
                    {
                        "type": "combined_keywords",
                        "text": ck_text,
                        "concept_id": concept_id,
                    },
                )
            )

        try:
            self.index.upsert(vectors=vectors)
            print(f"Upserted {len(vectors)} vectors for concept '{concept_id}'.")
        except Exception as e:
            print(f"Error upserting vectors to Pinecone: {e}")