from abc import ABC, abstractmethod


class VectorDatabase(ABC):
    @abstractmethod
    def upsert_concept_embeddings(self, embedding_data: dict):
        """
        Upsert concept embeddings into the vector database.

        Args:
            embedding_data (dict): Dictionary containing embeddings and metadata.
                - concept_id (str)
                - name (tuple): (text, embedding)
                - content (tuple): (text, embedding)
                - questions (list of tuples): [(text, embedding), ...]
                - keywords (list of tuples): [(text, embedding), ...]
                - combined_keywords (tuple or None): (text, embedding) or None
        """
        pass
