import re
import requests
from pathlib import Path
from logging import Logger
from bs4 import BeautifulSoup, Comment

from common.utils import DocType
from common.doc_base import Document


class DocWiki(Document):
    def __init__(self, source_str: str | Path, logger: Logger, output_dir: str = None):
        super().__init__(source_str, DocType.WIKI, logger, output_dir)

    def extract_plan_and_content(self, skip_if_exists: bool = False):
        super().generateOutputFile()

        # Get outputfile
        output_file = super().getOutputFile()

        # Skip if exists
        if skip_if_exists and output_file.exists():
            self.getLogger().info(f"\n\tFile already exists: {output_file.absolute()}")
            return {"title": str(self.getInput()), "content": "File already exists."}

        paper_data = self.extract_paper_data()
        paper_data = self.divide_into_chunks(paper_data)

        num_sections = len(paper_data.keys())
        min_required_sections = 3
        if num_sections < min_required_sections:
            error_msg = (
                f"\n\tInput: {self.getInput()}"
                f"\n\tInput document is too small. Found {num_sections} sections. We require at "
                f"least {min_required_sections} sections."
                f"\n\tSections Found: {list(paper_data.keys())}"
                f"\n\tSkipping this document and moving onto the next."
            )
            self.getLogger().error(error_msg)
            return {"title": str(self.getInput()), "content": error_msg}
        plan_json = self.generate_embeddings_plan_and_section_content(paper_data)

        # Write output
        self.writeOuputJson(plan_json)
        return plan_json

    def extract_paper_data(self):
        """Extracts the content from a Wikipedia URL into a dictionary. The keys are the header
        names + indicator of header level e.g. 'h2 Definitions'. The values are the content
        underneath each header tags.

        Only extracts content the user will see. Ignores hidden content and Contents list.

        Parameters
        ----------
        url : str
            The URL of the Wikipedia page to extract content from.

        Returns
        -------
        article_dict : dict
            A dictionary of the content extracted from the Wikipedia page.
        """
        url = self.getInput()
        if "wikipedia" not in url:
            raise ValueError("URL is not a Wikipedia URL. Received: " + url)

        # Wikipedia requires a User-Agent header to prevent blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(url, headers=headers)
        r.raise_for_status()  # Raise exception for HTTP errors
        html_content = r.text

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted tags
        for element in soup(["script", "style", "head", "title", "[document]"]):
            element.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
        found_tags = soup.find_all(tags)

        article_dict = {}
        reference_mapping = {}

        header_context = {}  # Track last seen headers: {1: h1_text, 2: h2_text, ...}

        for tag in found_tags:
            level = int(tag.name[1])  # Extract header level from h1, h2, ...
            header_context[level] = tag.get_text(strip=True)

            # Remove deeper level headers (h4+ if current is h3, etc.)
            for l in list(header_context.keys()):
                if l > level:
                    del header_context[l]

            # Build full key using header hierarchy
            key_parts = [header_context[lvl] for lvl in sorted(header_context) if 1 < lvl <= level]
            key = " - ".join(key_parts)
            if key == "":
                key = header_context[1]
            key = f"{tag.name} {key}"
            content = []
            next_tag = tag.find_next()
            # Look for next tags until the next header tag
            while next_tag and next_tag.name not in tags:
                # Reference section can contain both p and li tags
                if "reference" in key.lower() and next_tag.name in [
                    "p",
                    "li",
                ]:
                    ref_text = next_tag.get_text(strip=False)
                    content.append(ref_text)
                    # test = next_tag.next
                    # url = test.get("href","")
                    # while not url.startswith("http"):
                    #     test = test.next
                    #     url = test.get("href","")
                    # Extract citation number if present
                    tag_id = next_tag.attrs.get('id')
                    if tag_id and '-' in tag_id:
                        try:
                            cite_num = int(tag_id.split('-')[-1])
                            reference_mapping[cite_num] = ref_text
                        except (ValueError, IndexError):
                            pass  # Skip if ID format is unexpected

                elif next_tag.name == "p":
                    content.append(next_tag.get_text(strip=False))
                next_tag = next_tag.find_next()

            article_dict[key] = " ".join(content)

        # Clean up [edit] suffixes
        for key in list(
            article_dict.keys()
        ):  # Using list() to avoid changing the dictionary size during iteration
            if key.endswith("[edit]"):
                new_key = key.rsplit("[edit]", 1)[0]
                article_dict[new_key] = article_dict.pop(key)
        # Process references section
        ref_key = None
        article_dict["references"] = []
        for citation_num, ref_text in reference_mapping.items():
            if ref_text.strip():
                article_dict["references"].append(
                    {"resource_id": citation_num, "resource_description": ref_text}
                )
            else:
                self.getLogger().warning(
                    f"Empty reference text for citation number {citation_num}."
                )
        # Set references for the document
        self.setReferences(article_dict["references"])

        # Remove the references section from the main content
        article_dict.pop("h2 Contents", None)  # Some pages don't have Contents
        article_dict.pop("h2 References", None)  # Use pop() to avoid KeyError
        
        # Remove entries with empty values
        empty_keys = [key for key in article_dict.keys() if isinstance(article_dict[key], str) and article_dict[key].strip() == ""]
        for key in empty_keys:
            del article_dict[key]
        
            
            
        num_sections = len(article_dict.keys())

        self.getLogger().info(
            f"Successfully downloaded content from Wikipedia page {url}. "
            f"Extracted {num_sections} sections."
        )

        self.getLogger().info("EXTRACTION OVERVIEW:\nTitle:")

        # print for each key in article_dict, display key and frst 50 characters of value
        for key in article_dict.keys():
            self.getLogger().info(f"{key}: {article_dict[key][:80]}...")

        return article_dict
