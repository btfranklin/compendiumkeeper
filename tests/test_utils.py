from unittest.mock import patch, MagicMock

from compendiumkeeper.utils import (
    get_openai_api_key,
    get_embedding,
    slugify,
    generate_concept_id,
)


def test_slugify():
    assert slugify("Hello World!") == "hello_world"
    assert slugify("Python 3.12") == "python_3_12"
    assert slugify("   Leading and trailing   ") == "leading_and_trailing"


def test_generate_concept_id():
    id_value = generate_concept_id("My Topic", "My Concept")
    assert id_value == "my_topic_my_concept"


def test_get_openai_api_key():
    """
    We rely on the mock_env fixture from conftest.py for env var setup.
    """
    key = get_openai_api_key()
    assert key == "fake-openai-key"


@patch("compendiumkeeper.utils.OpenAI")
def test_get_embedding(mock_openai):
    """
    Test that get_embedding calls client.embeddings.create with correct arguments
    and returns the expected embedding array.
    """
    # Setup mock
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    mock_instance.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3])
    ]

    result = get_embedding("test text")
    assert result == [0.1, 0.2, 0.3]
    mock_openai.assert_called_once()
    mock_instance.embeddings.create.assert_called_once_with(
        model="text-embedding-ada-002", input="test text"
    )
