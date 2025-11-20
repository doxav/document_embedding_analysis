import logging
from time import time
from pathlib import Path
from typing import List
from common.doc_arxiv import DocArxiv
from common.doc_latext import DocLatex
from common.doc_wiki import DocWiki
from common.doc_patent import DocPatent
from common.doc_freshwiki import DocFreshWiki


# Function to process each type
def process_files(file_type: str, files: List, output_preference: str):
    for file in files:
        # initialize document
        if file_type == "l":
            doc = DocLatex(file, logging)
        elif file_type == "a":
            doc = DocArxiv(file, logging)
        elif file_type == "w":
            doc = DocWiki(file, logging)
        elif file_type == "p":
            doc = DocPatent(file, logging)
        elif file_type == "f":
            doc = DocFreshWiki(file, logging)
        else:
            continue

        # Measure begin time
        start = time()
        # try:
        #     doc.extract_plan_and_content()
        # except Exception as e:
        #     logging.error(f"Error occurred for {file}: {e}")
        doc.extract_plan_and_content()

        # Calculate time taken
        minutes, seconds = divmod(round(time() - start, 3), 60)
        elapsed = (
            f"{str(int(minutes)) + 'mins' if minutes else ''}"
            f" {str(round(seconds, 1)) + 's'}"
        )
        logging.info(
            f"\n\tSuccessfully extracted plan and content for {file}"
            f"\n\tWritten to file: {doc.getOutputFile()}"
            f"\n\tTime taken: {elapsed}"
        )

        # Check to continue or not
        if output_preference == "1" or output_preference == file_type:
            break


# Main function
def main():
    output_preference = input(
        "Generate for all (Enter), 1 for each type (1), or just an example of a given type (Latex: L, Arxiv: A, Wikipedia: W, Patent: P, FreshWiki: F): "
    ).lower()
    # output_preference = 'p'

    def _load_wikipedia_urls() -> List[str]:
        """
        If data/wiki_urls.txt exists, load URLs from there (one per line, '#' = comment).
        Otherwise, fall back to the original hard-coded list.
        """
        cfg_path = Path("data/wiki_urls.txt")
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                urls = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            logging.info(
                f"Loaded {len(urls)} Wikipedia URLs from {cfg_path}"
            )
            return urls

        # Fallback: original hard-coded list
        return [
            "https://en.wikipedia.org/wiki/Large_language_model",
            "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
            "https://en.wikipedia.org/wiki/Dual-phase_evolution",
            "https://en.wikipedia.org/wiki/Simulated_annealing",
            "https://en.wikipedia.org/wiki/Tessellation",
            "https://en.wikipedia.org/wiki/Climate_change",
            "https://en.wikipedia.org/wiki/DNA_nanotechnology",
            "https://en.wikipedia.org/wiki/Self-driving_car",
            "https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle",
            "https://en.wikipedia.org/wiki/2022%E2%80%932023_food_crises",
            "https://en.wikipedia.org/wiki/Economic_impacts_of_climate_change",
        ]

    # Mapping input to directories, URLs, and processing functions
    processing_map = {
        "l": Path("data/latex").glob("*.tex"),
        "a": Path("data/arxiv").glob("*"),
        "w": _load_wikipedia_urls(),
        "p": Path("data/patents").glob("*.txt"),
        "f": Path("data/freshwiki").glob("*.json"),
    }

    # Determine types to process
    types_to_process = (
        ["l", "a", "w", "p", "f"]
        if output_preference in ["", "1"]
        else [output_preference]
    )

    # Process each type
    for file_type in types_to_process:
        if file_type not in processing_map:
            continue
        files = processing_map[file_type]
        process_files(file_type, files, output_preference)


if __name__ == "__main__":
    # Set up logger
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    main()
    # file = Path('data/latex/A_Survey_on_Blood_Pressure_Measurement_Technologies_Addressing__Potential_Sources_of_Bias.tex')
    # doc = DocLatex(file, logging)
    # doc.extract_plan_and_content()
