# from main_ import compare_documents_sections, compare_documents_content
from common.test_utils import compare_documents_sections, compare_documents_content
import numpy as np


def is_approximately_equal(x, y, epsilon=1e-10):
    """Return True if two numbers are close in value. Cannot compare
    directly due to floating-point roundoff error."""
    return abs(x - y) < epsilon


def _test_same_document_comparison(file_path, doc_type, compare_func, compare_type):
    """Generic test for comparing a document to itself.
    
    Args:
        file_path: Path to the JSON file
        doc_type: Type of document (for error messages)
        compare_func: compare_documents_sections or compare_documents_content
        compare_type: 'plan' or 'content'
    """
    result = compare_func(file_path, file_path)
    
    # Test total similarity
    similarity_key = f"{compare_type}_total_similarity"
    similarity = result[similarity_key]
    
    assert is_approximately_equal(similarity["rouge_L_similarity"], 1.0), \
        f"{doc_type} {compare_type}: ROUGE-L should be 1.0 for identical documents, got {similarity['rouge_L_similarity']}"
    assert is_approximately_equal(similarity["embedding1_cosine_similarity"], 1.0), \
        f"{doc_type} {compare_type}: Embedding1 cosine should be 1.0 for identical documents, got {similarity['embedding1_cosine_similarity']}"
    assert is_approximately_equal(similarity["embedding2_cosine_similarity"], 1.0), \
        f"{doc_type} {compare_type}: Embedding2 cosine should be 1.0 for identical documents, got {similarity['embedding2_cosine_similarity']}"
    assert similarity["mauve_similarity"] >= 0.7, \
        f"{doc_type} {compare_type}: MAUVE should be >= 0.7 for identical documents, got {similarity['mauve_similarity']}"
    
    # Test per-section similarities
    bysection_key = f"{compare_type}_bysection_similarity"
    bysection = result[bysection_key]
    assert len(bysection) > 0, f"{doc_type}: Should have per-section {compare_type} similarities"
    
    for section in bysection:
        assert is_approximately_equal(section["rouge_L_similarity"], 1.0), \
            f"{doc_type} {compare_type} section {section['section_id']}: ROUGE-L should be 1.0, got {section['rouge_L_similarity']}"
        assert is_approximately_equal(section["embedding1_cosine_similarity"], 1.0), \
            f"{doc_type} {compare_type} section {section['section_id']}: embedding1 should be 1.0, got {section['embedding1_cosine_similarity']}"
        assert is_approximately_equal(section["embedding2_cosine_similarity"], 1.0), \
            f"{doc_type} {compare_type} section {section['section_id']}: embedding2 should be 1.0, got {section['embedding2_cosine_similarity']}"
        assert section["mauve_similarity"] >= 0.7, \
            f"{doc_type} {compare_type} section {section['section_id']}: MAUVE should be >= 0.7, got {section['mauve_similarity']}"
    
    # Test bibliography coverage (only for sections with references)
    sections_with_refs = [s for s in bysection if s["bibliography_coverage1"] > 0]
    if len(sections_with_refs) > 0:
        for section in sections_with_refs:
            assert section["bibliography_coverage1"] >= 0.9, \
                f"{doc_type} {compare_type} section {section['section_id']} with refs: bib coverage1 should be >= 0.9, got {section['bibliography_coverage1']}"
            assert section["bibliography_coverage2"] >= 0.9, \
                f"{doc_type} {compare_type} section {section['section_id']} with refs: bib coverage2 should be >= 0.9, got {section['bibliography_coverage2']}"
    
    # Global bibliography coverage (only if document has references)
    if result["bibliography_coverage1"] > 0:  # Has references
        assert result["bibliography_coverage1"] >= 0.9, \
            f"{doc_type} {compare_type}: Global bibliography coverage1 should be >= 0.9 for identical documents, got {result['bibliography_coverage1']}"
        assert result["bibliography_coverage2"] >= 0.9, \
            f"{doc_type} {compare_type}: Global bibliography coverage2 should be >= 0.9 for identical documents, got {result['bibliography_coverage2']}"


def _test_different_documents_comparison(file_path1, file_path2, doc_type, compare_func, compare_type):
    """Generic test for comparing two different documents.
    
    Args:
        file_path1: Path to first JSON file
        file_path2: Path to second JSON file
        doc_type: Type of document (for error messages)
        compare_func: compare_documents_sections or compare_documents_content
        compare_type: 'plan' or 'content'
    """
    result = compare_func(file_path1, file_path2)
    
    # Test total similarity - should be lower
    similarity_key = f"{compare_type}_total_similarity"
    for value in result[similarity_key].values():
        assert not is_approximately_equal(value, 1.0), \
            f"{doc_type} {compare_type}: Different documents should not have perfect similarity, got {value}"
    
    # Per-section similarities - at least some should be lower
    bysection_key = f"{compare_type}_bysection_similarity"
    bysection = result[bysection_key]
    assert len(bysection) > 0, f"{doc_type}: Should have per-section {compare_type} similarities"
    
    non_perfect_sections = sum(1 for s in bysection if not is_approximately_equal(s["rouge_L_similarity"], 1.0))
    assert non_perfect_sections > 0, f"{doc_type} {compare_type}: Different documents should have non-perfect section scores"
    
    # Global bibliography coverage should be lower (only if documents have references)
    if result["bibliography_coverage1"] > 0:  # Has references
        assert result["bibliography_coverage1"] < 0.9, \
            f"{doc_type} {compare_type}: Global bibliography coverage1 should be < 0.9 for different documents, got {result['bibliography_coverage1']}"
        assert result["bibliography_coverage2"] < 0.9, \
            f"{doc_type} {compare_type}: Global bibliography coverage2 should be < 0.9 for different documents, got {result['bibliography_coverage2']}"
        
        # Average per-section bibliography coverage (for sections with refs)
        sections_with_refs = [s for s in bysection if s["bibliography_coverage1"] > 0]
        if len(sections_with_refs) > 0:
            avg_bib_cov1 = np.mean([s["bibliography_coverage1"] for s in sections_with_refs])
            avg_bib_cov2 = np.mean([s["bibliography_coverage2"] for s in sections_with_refs])
            assert avg_bib_cov1 < 0.9, \
                f"{doc_type} {compare_type}: Average per-section bib coverage1 should be < 0.9 for different documents, got {avg_bib_cov1}"
            assert avg_bib_cov2 < 0.9, \
                f"{doc_type} {compare_type}: Average per-section bib coverage2 should be < 0.9 for different documents, got {avg_bib_cov2}"


# Wikipedia Tests
def test_compare_documents_plans_same_document():
    _test_same_document_comparison(
        "output/wikipedia/Dual-phase evolution.json",
        "Wikipedia",
        compare_documents_sections,
        "plan"
    )


def test_compare_documents_sections_same_document():
    _test_same_document_comparison(
        "output/wikipedia/Dual-phase evolution.json",
        "Wikipedia",
        compare_documents_content,
        "content"
    )


def test_compare_documents_plans_different_documents():
    _test_different_documents_comparison(
        "output/wikipedia/Dual-phase evolution.json",
        "output/wikipedia/Climate change.json",
        "Wikipedia",
        compare_documents_sections,
        "plan"
    )


def test_compare_documents_sections_different_documents():
    _test_different_documents_comparison(
        "output/wikipedia/Dual-phase evolution.json",
        "output/wikipedia/Climate change.json",
        "Wikipedia",
        compare_documents_content,
        "content"
    )


# LaTeX Tests
def test_latex_plans_same_document():
    _test_same_document_comparison(
        "output/latex/A_survey_on_algebraic_dilatations_with_embeddings.json",
        "LaTeX",
        compare_documents_sections,
        "plan"
    )


def test_latex_content_same_document():
    _test_same_document_comparison(
        "output/latex/A_survey_on_algebraic_dilatations_with_embeddings.json",
        "LaTeX",
        compare_documents_content,
        "content"
    )


def test_latex_plans_different_documents():
    _test_different_documents_comparison(
        "output/latex/A_survey_on_algebraic_dilatations_with_embeddings.json",
        "output/latex/A_Survey_on_Blood_Pressure_Measurement_Technologies_Addressing__Potential_Sources_of_Bias_with_embeddings.json",
        "LaTeX",
        compare_documents_sections,
        "plan"
    )


def test_latex_content_different_documents():
    _test_different_documents_comparison(
        "output/latex/A_survey_on_algebraic_dilatations_with_embeddings.json",
        "output/latex/A_Survey_on_Blood_Pressure_Measurement_Technologies_Addressing__Potential_Sources_of_Bias_with_embeddings.json",
        "LaTeX",
        compare_documents_content,
        "content"
    )


# Patent Tests
def test_patent_plans_same_document():
    _test_same_document_comparison(
        "output/patent/APPARATUS FOR THE MEASUREMENT OF ATRIAL PRESSURE.json",
        "Patent",
        compare_documents_sections,
        "plan"
    )


def test_patent_content_same_document():
    _test_same_document_comparison(
        "output/patent/APPARATUS FOR THE MEASUREMENT OF ATRIAL PRESSURE.json",
        "Patent",
        compare_documents_content,
        "content"
    )


def test_patent_plans_different_documents():
    _test_different_documents_comparison(
        "output/patent/APPARATUS FOR THE MEASUREMENT OF ATRIAL PRESSURE.json",
        "output/patent/A PROCESS FOR REGENERATING SPENT FLUIDIZED CATALYTIC CRACKING CATALYST.json",
        "Patent",
        compare_documents_sections,
        "plan"
    )


def test_patent_content_different_documents():
    _test_different_documents_comparison(
        "output/patent/APPARATUS FOR THE MEASUREMENT OF ATRIAL PRESSURE.json",
        "output/patent/A PROCESS FOR REGENERATING SPENT FLUIDIZED CATALYTIC CRACKING CATALYST.json",
        "Patent",
        compare_documents_content,
        "content"
    )


# ArXiv Tests
def test_arxiv_plans_same_document():
    _test_same_document_comparison(
        "output/arxiv/A survey on algebraic dilatations.json",
        "ArXiv",
        compare_documents_sections,
        "plan"
    )


def test_arxiv_content_same_document():
    _test_same_document_comparison(
        "output/arxiv/A survey on algebraic dilatations.json",
        "ArXiv",
        compare_documents_content,
        "content"
    )


def test_arxiv_plans_different_documents():
    _test_different_documents_comparison(
        "output/arxiv/A survey on algebraic dilatations.json",
        "output/arxiv/A Survey on Blood Pressure Measurement Technologies Addressing  Potential Sources of Bias.json",
        "ArXiv",
        compare_documents_sections,
        "plan"
    )


def test_arxiv_content_different_documents():
    _test_different_documents_comparison(
        "output/arxiv/A survey on algebraic dilatations.json",
        "output/arxiv/A Survey on Blood Pressure Measurement Technologies Addressing  Potential Sources of Bias.json",
        "ArXiv",
        compare_documents_content,
        "content"
    )


def run_tests():
    # Wikipedia
    test_compare_documents_plans_same_document()
    test_compare_documents_sections_same_document()
    test_compare_documents_plans_different_documents()
    test_compare_documents_sections_different_documents()
    
    # LaTeX
    test_latex_plans_same_document()
    test_latex_content_same_document()
    test_latex_plans_different_documents()
    test_latex_content_different_documents()
    
    # Patent
    test_patent_plans_same_document()
    test_patent_content_same_document()
    test_patent_plans_different_documents()
    test_patent_content_different_documents()
    
    # ArXiv
    test_arxiv_plans_same_document()
    test_arxiv_content_same_document()
    test_arxiv_plans_different_documents()
    test_arxiv_content_different_documents()
    
    print()
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
