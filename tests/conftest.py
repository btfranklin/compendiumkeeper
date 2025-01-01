import os
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_env():
    """
    Automatically mock environment variables for all tests, unless overridden.
    """
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "fake-openai-key",
            "PINECONE_API_KEY": "fake-pinecone-key",
            "PINECONE_ENVIRONMENT": "us-east-1-aws",
        },
        clear=True,
    ):
        yield
