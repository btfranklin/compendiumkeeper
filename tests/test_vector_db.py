from unittest.mock import patch, MagicMock, ANY, call

from compendiumkeeper.vector_db.pinecone_db import PineconeDB


@patch("compendiumkeeper.vector_db.pinecone_db.Pinecone")
def test_pinecone_db_init__index_not_found(mock_pinecone):
    """
    Scenario: The index does NOT exist initially.
    We expect:
     - create_index() is called
     - describe_index_stats(index_name) is called
     - .Index(host=...) is called exactly once for final usage
     - No calls to describe_index() because the index didn't exist
    """
    mock_client_instance = MagicMock()
    mock_pinecone.return_value = mock_client_instance

    # The index is not in the list => creation block triggers
    mock_client_instance.list_indexes.return_value = []

    PineconeDB(index_name="testindex")

    # Confirm Pinecone() was constructed once
    mock_pinecone.assert_called_once()

    # Because the index isn't in the list, we do create_index
    mock_client_instance.create_index.assert_called_once_with(
        name="testindex", dimension=1536, metric="cosine", spec=ANY
    )

    # The code next calls describe_index_stats("testindex") to get the host
    mock_client_instance.describe_index_stats.assert_called_once_with("testindex")

    # Because the index didn't exist, we do NOT call describe_index
    mock_client_instance.describe_index.assert_not_called()

    # Finally, .Index(host=...) is called exactly once
    mock_client_instance.Index.assert_called_once_with(host=ANY)


@patch("compendiumkeeper.vector_db.pinecone_db.Pinecone")
def test_pinecone_db_init__index_exists(mock_pinecone):
    """
    Scenario: The index already exists.
    We expect:
     - The code to call describe_index(index_name) and do .Index(host=...).delete(...)
     - Then skip create_index
     - No call to describe_index_stats (since that's for new indexes)
     - Finally .Index(host=...) again for final usage
    => That yields TWO calls to .Index(host=...) total.
    """
    mock_client_instance = MagicMock()
    mock_pinecone.return_value = mock_client_instance

    # The index is found => clearing block triggers
    mock_client_instance.list_indexes.return_value = ["testindex"]

    PineconeDB(index_name="testindex")

    # We skip creation
    mock_client_instance.create_index.assert_not_called()

    # The code calls describe_index("testindex") but not describe_index_stats
    mock_client_instance.describe_index.assert_called_once_with("testindex")
    mock_client_instance.describe_index_stats.assert_not_called()

    # The first .Index(...) call is for deleting all vectors
    # The second .Index(...) call is for final usage
    # We'll just confirm that somewhere in those calls we have
    # `.delete(delete_all=True)` being invoked on the mock.
    assert mock_client_instance.Index.call_count == 2
    index_mock_calls = mock_client_instance.Index.mock_calls
    assert (
        call().delete(delete_all=True) in index_mock_calls
    ), "Expected a call to .delete(delete_all=True)"


@patch("compendiumkeeper.vector_db.pinecone_db.Pinecone")
def test_pinecone_db_upsert(mock_pinecone):
    """
    Test upsert_concept_embeddings to ensure correct data is upserted.
    We'll assume the index already exists for this scenario.
    """
    mock_client_instance = MagicMock()
    mock_pinecone.return_value = mock_client_instance

    # Index found => clearing block triggers, then final usage
    mock_client_instance.list_indexes.return_value = ["testindex"]

    db = PineconeDB(index_name="testindex")

    # Because "testindex" was found in the list, we skip create_index
    mock_client_instance.create_index.assert_not_called()

    # We also see the code calls describe_index("testindex") and does a first .Index for deleting
    # and a second .Index for final usage
    assert mock_client_instance.Index.call_count == 2

    # The second .Index(...) instance is the one used for upsert
    mock_index = mock_client_instance.Index.return_value

    embedding_data = {
        "concept_id": "topic_concept",
        "name": ("Concept Name", [0.1, 0.2]),
        "content": ("Concept Content", [0.3, 0.4]),
        "questions": [("Q1", [0.5]), ("Q2", [0.6])],
        "keywords": [("keyword1", [0.7])],
        "combined_keywords": ("keyword1", [0.8]),
    }
    db.upsert_concept_embeddings(embedding_data)

    # upsert should be called exactly once
    mock_index.upsert.assert_called_once()

    # Because the code calls upsert(vectors=vectors)
    all_upsert_calls = mock_index.upsert.call_args_list
    assert len(all_upsert_calls) == 1
    upsert_args, upsert_kwargs = all_upsert_calls[0]

    # If the code calls upsert(vectors=...), that means upsert_kwargs["vectors"] has the data
    if "vectors" in upsert_kwargs:
        vectors_list = upsert_kwargs["vectors"]
    else:
        # If code used a positional argument, it'd be in upsert_args
        vectors_list = upsert_args[0]

    assert (
        len(vectors_list) == 6
    ), "Expected 6 total vectors for concept's name, content, 2 questions, 1 keyword, 1 combined_keywords"

    for vector_id, vector_emb, metadata in vectors_list:
        assert metadata["concept_id"] == "topic_concept"
