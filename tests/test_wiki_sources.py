from bs4 import BeautifulSoup

from common.doc_wiki import DocWiki


def test_wikipedia_external_reference_url_prefers_absolute_source_links():
    tag = BeautifulSoup(
        '<li id="cite_note-1"><a href="/wiki/Internal">Internal</a><a href="//example.org/source">Source</a></li>',
        "html.parser",
    ).li

    assert DocWiki._external_reference_url(tag) == "https://example.org/source"
