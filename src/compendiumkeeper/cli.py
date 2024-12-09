import click
import os
from dotenv import load_dotenv

from compendiumkeeper.indexer import index_compendium


@click.group()
def main():
    """Compendium Keeper CLI"""
    pass


@main.command("index")
@click.option(
    "--compendium-file", "-c", required=True, help="Path to the Compendium pickle file."
)
def index_cmd(compendium_file):
    """
    Index a Compendium into a vector database.
    """
    # Load environment variables from the .env file automatically
    load_dotenv()

    # Derive index name from compendium file
    # e.g., synthetic_technocracy_2024-12-01.compendium.pickle -> synthetic_technocracy_2024-12-01
    base_name = os.path.basename(compendium_file)
    root, ext = os.path.splitext(base_name)
    if root.endswith(".compendium"):
        index_name = root[: -len(".compendium")]
    else:
        index_name = root

    # Pinecone requires the name to be lowercase alphanumeric or '-'.
    # So we will:
    # 1. Lowercase the entire name
    # 2. Replace underscores with dashes
    index_name = index_name.lower().replace("_", "-")

    try:
        index_compendium(
            compendium_file=compendium_file,
            vector_db_type="pinecone",
            index_name=index_name,
        )
        click.secho("Indexing complete!", fg="green")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        SystemExit(1)


if __name__ == "__main__":
    main()
