import pickle
import xml.etree.ElementTree as ET

from compendiumscribe.model import Domain, Topic, Concept
from compendiumkeeper.utils import get_embedding_data
from compendiumkeeper.vector_db.pinecone_db import PineconeDB


def index_compendium(compendium_file: str, vector_db_type: str, index_name: str):
    """
    Load a Compendium from either an XML file or a pickle file,
    then index its contents into the specified vector DB index.
    """
    domain = None
    if compendium_file.endswith(".compendium.pickle"):
        domain = load_domain_from_pickle(compendium_file)
    elif compendium_file.endswith(".compendium.xml"):
        domain = load_domain_from_xml(compendium_file)
    else:
        raise RuntimeError(
            f"Unknown file format for '{compendium_file}'. "
            "Expected .compendium.pickle or .compendium.xml"
        )

    # Initialize the vector database client
    if vector_db_type == "pinecone":
        vector_db = PineconeDB(index_name=index_name)
    else:
        raise RuntimeError(f"Unsupported vector DB: {vector_db_type}")

    total_concepts = 0
    for topic in domain.topics:
        for concept in topic.concepts:
            embedding_data = get_embedding_data(
                concept, topic_summary=topic.topic_summary, topic_name=topic.name
            )
            vector_db.upsert_concept_embeddings(embedding_data)
            total_concepts += 1

    print(
        f"Indexed {total_concepts} concepts from domain '{domain.name}' into index '{index_name}'."
    )


def load_domain_from_pickle(filepath: str) -> Domain:
    """Load a Domain object from a pickle file."""
    try:
        with open(filepath, "rb") as f:
            domain = pickle.load(f)
        return domain
    except Exception as e:
        raise RuntimeError(f"Error loading domain from pickle file '{filepath}': {e}")


def load_domain_from_xml(filepath: str) -> Domain:
    """
    Load a Domain object from an XML file.
    Expects the root element to be <domain>.
    """
    try:
        tree = ET.parse(filepath)
        domain_elem = tree.getroot()
        if domain_elem.tag != "domain":
            raise ValueError(
                f"Root element should be <domain>, got <{domain_elem.tag}>"
            )

        # Parse domain attributes
        domain_name = domain_elem.attrib.get("name", "")
        domain_summary_elem = domain_elem.find("summary")
        domain_summary = (
            domain_summary_elem.text if domain_summary_elem is not None else ""
        )

        # Construct the Domain object
        domain = Domain(name=domain_name, summary=domain_summary, topics=[])

        # Parse <topic> elements
        for topic_elem in domain_elem.findall("topic"):
            topic = parse_topic_xml(topic_elem)
            domain.topics.append(topic)

        return domain

    except Exception as e:
        raise RuntimeError(f"Error loading domain from XML file '{filepath}': {e}")


def parse_topic_xml(topic_elem: ET.Element) -> Topic:
    """
    Parse a <topic> element into a Topic object.
    """
    topic_name = topic_elem.attrib.get("name", "")
    topic_summary_elem = topic_elem.find("topic_summary")
    topic_summary = topic_summary_elem.text if topic_summary_elem is not None else ""

    topic = Topic(name=topic_name, topic_summary=topic_summary, concepts=[])

    # Concepts
    concepts_parent = topic_elem.find("concepts")
    if concepts_parent is not None:
        for concept_elem in concepts_parent.findall("concept"):
            concept = parse_concept_xml(concept_elem)
            topic.concepts.append(concept)

    return topic


def parse_concept_xml(concept_elem: ET.Element) -> Concept:
    """
    Parse a <concept> element into a Concept object.
    """
    concept_name = concept_elem.attrib.get("name", "")
    concept = Concept(name=concept_name)

    # questions
    questions_elem = concept_elem.find("questions")
    if questions_elem is not None:
        for question_elem in questions_elem.findall("question"):
            if question_elem.text:
                concept.questions.append(question_elem.text.strip())

    # keywords
    keywords_elem = concept_elem.find("keywords")
    if keywords_elem is not None:
        for keyword_elem in keywords_elem.findall("keyword"):
            if keyword_elem.text:
                concept.keywords.append(keyword_elem.text.strip())

    # prerequisites
    prereqs_elem = concept_elem.find("prerequisites")
    if prereqs_elem is not None:
        for prereq_elem in prereqs_elem.findall("prerequisite"):
            if prereq_elem.text:
                concept.prerequisites.append(prereq_elem.text.strip())

    # content
    content_elem = concept_elem.find("content")
    if content_elem is not None and content_elem.text:
        concept.content = content_elem.text.strip()

    return concept
