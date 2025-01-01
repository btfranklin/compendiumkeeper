import pytest
import pickle
import shutil
import xml.etree.ElementTree as ET
from unittest.mock import patch

from compendiumscribe.model import Domain, Topic, Concept
from compendiumkeeper.indexer import (
    index_compendium,
    load_domain_from_pickle,
    load_domain_from_xml,
)


@pytest.fixture
def temp_dir(tmp_path):
    """
    Pytest fixture for creating a temporary directory to store test files.
    """
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_load_domain_from_pickle(temp_dir):
    """
    Test that load_domain_from_pickle correctly loads a Domain from a pickle.
    """
    domain = Domain(name="Test Domain", summary="Just a test")
    domain.topics.append(Topic(name="Test Topic", topic_summary="Topic Summary"))

    pickle_file = temp_dir / "test.compendium.pickle"
    with open(pickle_file, "wb") as f:
        pickle.dump(domain, f)

    loaded_domain = load_domain_from_pickle(str(pickle_file))
    assert loaded_domain.name == "Test Domain"
    assert loaded_domain.summary == "Just a test"
    assert len(loaded_domain.topics) == 1
    assert loaded_domain.topics[0].name == "Test Topic"


def test_load_domain_from_xml(temp_dir):
    """
    Test that load_domain_from_xml correctly loads a Domain from an XML file.
    """
    xml_file = temp_dir / "test.compendium.xml"

    # Manually build minimal XML
    domain_elem = ET.Element("domain", attrib={"name": "XML Domain"})
    summary_elem = ET.SubElement(domain_elem, "summary")
    summary_elem.text = "XML summary"

    topic_elem = ET.SubElement(domain_elem, "topic", attrib={"name": "XML Topic"})
    topic_summary = ET.SubElement(topic_elem, "topic_summary")
    topic_summary.text = "XML Topic Summary"

    concepts = ET.SubElement(topic_elem, "concepts")
    concept_elem = ET.SubElement(concepts, "concept", attrib={"name": "XML Concept"})
    content_elem = ET.SubElement(concept_elem, "content")
    content_elem.text = "XML Concept Content"

    tree = ET.ElementTree(domain_elem)
    tree.write(xml_file)

    loaded_domain = load_domain_from_xml(str(xml_file))
    assert loaded_domain.name == "XML Domain"
    assert loaded_domain.summary == "XML summary"
    assert len(loaded_domain.topics) == 1
    assert loaded_domain.topics[0].name == "XML Topic"
    assert len(loaded_domain.topics[0].concepts) == 1
    assert loaded_domain.topics[0].concepts[0].content == "XML Concept Content"


@patch("compendiumkeeper.indexer.load_domain_from_pickle")
@patch("compendiumkeeper.indexer.PineconeDB")
@patch("compendiumkeeper.utils.get_embedding", return_value=[0.1, 0.2, 0.3])
def test_index_compendium_pickle(
    mock_get_embedding, mock_pinecone, mock_load_pickle, temp_dir
):
    """
    Test index_compendium with a pickle file.
    Ensures that after loading the Domain, we upsert the correct number of concepts.
    """
    # Setup: create a domain with 2 topics, each with 1 concept
    domain = Domain(name="Pickle Domain")
    topic1 = Topic(name="Topic1")
    topic1.concepts.append(Concept(name="Concept1"))
    domain.topics.append(topic1)
    topic2 = Topic(name="Topic2")
    topic2.concepts.append(Concept(name="Concept2"))
    domain.topics.append(topic2)

    mock_load_pickle.return_value = domain

    pickle_file = temp_dir / "test.compendium.pickle"
    pickle_file.touch()  # create an empty file

    index_compendium(str(pickle_file), vector_db_type="pinecone", index_name="my_index")

    # Verify the domain was loaded from pickle
    mock_load_pickle.assert_called_once_with(str(pickle_file))

    # PineconeDB should have been instantiated with index_name="my_index"
    mock_pinecone.assert_called_once_with(index_name="my_index")

    # The mock's upsert_concept_embeddings should be called once for each concept
    pinecone_instance = mock_pinecone.return_value
    assert pinecone_instance.upsert_concept_embeddings.call_count == 2

    # Ensure get_embedding was called for each relevant chunk
    # We have 2 concepts, each embedding name + content, plus no. of questions, etc.
    # This is an approximation check.
    # More rigorous approach:
    #   you could check how many times each concept part is embedded
    embed_call_count = mock_get_embedding.call_count
    assert embed_call_count > 0  # ensures no real call to OpenAI was made


@patch("compendiumkeeper.indexer.load_domain_from_xml")
@patch("compendiumkeeper.indexer.PineconeDB")
@patch("compendiumkeeper.utils.get_embedding", return_value=[0.99, 0.98])
def test_index_compendium_xml(
    mock_get_embedding, mock_pinecone, mock_load_xml, temp_dir
):
    """
    Test index_compendium with an XML file.
    """
    # Setup: create a domain with 1 topic, 2 concepts
    domain = Domain(name="XML Domain")
    topic = Topic(name="XML Topic")
    topic.concepts.append(Concept(name="C1"))
    topic.concepts.append(Concept(name="C2"))
    domain.topics.append(topic)

    mock_load_xml.return_value = domain

    xml_file = temp_dir / "test.compendium.xml"
    xml_file.touch()

    index_compendium(
        str(xml_file), vector_db_type="pinecone", index_name="shared_index"
    )

    # Verify domain was loaded from XML
    mock_load_xml.assert_called_once_with(str(xml_file))

    # PineconeDB should have been instantiated with index_name="shared_index"
    mock_pinecone.assert_called_once_with(index_name="shared_index")

    pinecone_instance = mock_pinecone.return_value
    # 1 topic x 2 concepts => 2 calls to upsert
    assert pinecone_instance.upsert_concept_embeddings.call_count == 2

    # Also ensure get_embedding was called
    assert mock_get_embedding.call_count > 0
