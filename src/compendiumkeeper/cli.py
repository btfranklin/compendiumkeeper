# compendiumkeeper/cli.py
import click
import sys
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
    # e.g., cell_biology_2024-12-05.compendium.pickle -> cell_biology_2024-12-05
    base_name = os.path.basename(
        compendium_file
    )  # e.g. cell_biology_2024-12-05.compendium.pickle
    root, ext = os.path.splitext(
        base_name
    )  # root=cell_biology_2024-12-05.compendium, ext=.pickle
    if root.endswith(".compendium"):
        index_name = root[: -len(".compendium")]
    else:
        # If for some reason it doesn't match, just use the root
        index_name = root

    try:
        index_compendium(
            compendium_file=compendium_file,
            vector_db_type="pinecone",
            index_name=index_name,
        )
        click.secho("Indexing complete!", fg="green")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
