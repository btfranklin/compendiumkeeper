import click
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
@click.option(
    "--index-name", "-i", required=True, help="Name of the vector database index."
)
def index_cmd(compendium_file, index_name):
    """
    Index a Compendium into a vector database.
    """
    # Load environment variables from the .env file automatically
    load_dotenv()

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
