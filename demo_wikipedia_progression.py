import json
import sys
from pathlib import Path
from textwrap import dedent
from common.doc_eval import evaluate_document
import numpy as np

# Helper to print scores nicely
def print_scores(stage_name, scores):
    dea = scores.get("dea_evaluation_scores") or {}
    prom = scores.get("prometheus_scores") or {}
    wh = scores.get("writehere_scores") or {}
    art = scores.get("article_metrics") or {}

    print(f"\n=== {stage_name} ===")
    metrics = {
        "Plan Similarity": dea.get("plan_embedding_similarity"),
        "Plan Titles Sim": dea.get("plan_titles_embedding_similarity"),
        "Plan Contents Sim": dea.get("plan_contents_embedding_similarity"),
        "Resources Sim": dea.get("plan_resources_embedding_similarity"),
        "Sections Ratio": dea.get("sections_count_ratio_to_target"),
        "Content Len Ratio": dea.get("content_length_ratio_to_target"),
        "Resources Coverage": dea.get("resources_citation_coverage_score"),
        "Resources Count Ratio": dea.get("resources_count_ratio_to_target"),
    }
    for k, v in metrics.items():
        val = v if v is not None else "N/A"
        if isinstance(val, (float, np.floating)):
            print(f"  {k:<20}: {val:.4f}")
        else:
            print(f"  {k:<20}: {val}")

    if prom:
        print("\n  [Prometheus]")
        for k, v in prom.items():
            print(f"  {k:<20}: {v:.4f}")

    if wh:
        print("\n  [WriteHere]")
        for k, v in wh.items():
            print(f"  {k:<20}: {v:.4f}")

    if art:
        print("\n  [Article Quality]")
        if "entity_recall" in art:
            print(f"  {'Entity Recall':<20}: {art['entity_recall']:.4f}")
        if "citation_count" in art:
            print(f"  {'Citation Count':<20}: {art['citation_count']}")
        if "rouge_scores" in art:
            print(f"  {'ROUGE-L F1':<20}: {art['rouge_scores']['rouge-l']['f']:.4f}")

def main():
    # Configuration
    USE_ENHANCED_METRICS = True

    # Load solution
    solution_path = Path('output/wikipedia/Transformer (machine learning model).json')
    if not solution_path.exists():
        print(f"Error: Solution file not found at {solution_path}")
        sys.exit(1)
    
    solution = json.loads(solution_path.read_text())
    print(f"Loaded solution: {solution.get('title')}")

    base_latex = dedent(r'''
    \documentclass{article}
    \begin{document}
    \title{Transformer (deep learning architecture)}
    \maketitle
    ''')

    # Stage 1: Irrelevant Content
    # Content is about baking, completely unrelated to Transformers.
    stage1_content = base_latex + dedent(r'''
    \section{Introduction}
    Baking a cake requires flour, sugar, eggs, and butter. 
    Mix the ingredients thoroughly and bake at 350 degrees.
    \section{Frosting}
    Buttercream frosting is made with butter and powdered sugar.
    \end{document}
    ''')

    # Stage 2: Relevant Content, Poor Plan
    # Content is relevant (about Transformers) but structure is just one big section "Overview".
    stage2_content = base_latex + dedent(r'''
    \section{Overview}
    A transformer is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism.
    Transformers were proposed in 2017 as an attention-only alternative to RNNs.
    Teacher forcing, label smoothing, and warmup schedules stabilize optimization.
    Stacked self-attention and feed-forward layers with residual connections and layer norm; decoders use causal masks.
    Translation, summarization, speech, and modern LLMs.
    \end{document}
    ''')

    # Stage 3: Relevant Content, Good Plan
    # Structure matches the solution (History, Training, Architecture, Applications).
    stage3_content = base_latex + dedent(r'''
    \section{History}
    Transformers were proposed in 2017 as an attention-only alternative to RNNs.
    \section{Training}
    Teacher forcing, label smoothing, and warmup schedules stabilize optimization.
    \section{Architecture}
    Stacked self-attention and feed-forward layers with residual connections and layer norm; decoders use causal masks.
    \section{Applications}
    Translation, summarization, speech, and modern LLMs.
    \end{document}
    ''')

    # Stage 4: Relevant Content, Good Plan, Bibliography
    # Adds citations and bibliography.
    stage4_content = base_latex + dedent(r'''
    \section{History}
    Transformers originated with \cite{vaswani2017attention}, replacing recurrence with attention and positional encodings.
    \section{Training}
    Adam with warmup, dropout, and label smoothing enable stable pretraining; fine-tuning adapts to downstream tasks.
    \section{Architecture}
    Multi-head self-attention, feed-forward blocks, residuals, and normalization; decoder uses masked attention for autoregressive outputs.
    \section{Applications}
    Dominant in translation, summarization, vision-language systems, protein folding, and GPT-style LLMs.
    \begin{thebibliography}{9}
    \bibitem{vaswani2017attention} Vaswani et al. Attention Is All You Need. NeurIPS 2017.
    \bibitem{bert2018} Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018.
    \end{thebibliography}
    \end{document}
    ''')

    # Stage 5: Relevant Content, Good Plan, Rich Bibliography
    # Adds many citations and bibliography.
    stage5_content = base_latex + dedent(r'''
    \section{History}
    Transformers originated with \cite{vaswani2017attention}, replacing recurrence with attention and positional encodings.
    \section{Training}
    Adam with warmup, dropout, and label smoothing enable stable pretraining; fine-tuning adapts to downstream tasks.
    \section{Architecture}
    Multi-head self-attention, feed-forward blocks, residuals, and normalization; decoder uses masked attention for autoregressive outputs.
    \section{Applications}
    Dominant in translation, summarization, vision-language systems, protein folding, and GPT-style LLMs.
    \begin{thebibliography}{99}
    \bibitem{vaswani2017attention} Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N; Kaiser, \u0141ukasz; Polosukhin, Illia (2017). "Attention is All you Need" (PDF). Advances in Neural Information Processing Systems. 30. Curran Associates, Inc.
    \bibitem{hochreiter1997} Hochreiter, Sepp; Schmidhuber, J\u00fcrgen (1 November 1997). "Long Short-Term Memory". Neural Computation. 9 (8): 1735\u20131780.
    \bibitem{openai2019} "Better Language Models and Their Implications". OpenAI. 2019-02-14.
    \bibitem{bahdanau2014} Bahdanau; Cho, Kyunghyun; Bengio, Yoshua (September 1, 2014). "Neural Machine Translation by Jointly Learning to Align and Translate". arXiv:1409.0473 [cs.CL].
    \bibitem{luong2015} Luong, Minh-Thang; Pham, Hieu; Manning, Christopher D. (August 17, 2015). "Effective Approaches to Attention-based Neural Machine Translation". arXiv:1508.04025 [cs.CL].
    \bibitem{chen2021} Chen, Lili; Lu, Kevin; Rajeswaran, Aravind; Lee, Kimin; Grover, Aditya; Laskin, Michael; Abbeel, Pieter; Srinivas, Aravind; Mordatch, Igor (2021-06-24), Decision Transformer: Reinforcement Learning via Sequence Modeling, arXiv:2106.01345
    \bibitem{parisotto2020} Parisotto, Emilio; Song, Francis; Rae, Jack; Pascanu, Razvan; Gulcehre, Caglar; Jayakumar, Siddhant; Jaderberg, Max; Kaufman, Rapha\u00ebl Lopez; Clark, Aidan; Noury, Seb; Botvinick, Matthew; Heess, Nicolas; Hadsell, Raia (2020-11-21). "Stabilizing Transformers for Reinforcement Learning". Proceedings of the 37th International Conference on Machine Learning. PMLR: 7487\u20137498.
    \bibitem{radford2022} Radford, Alec; Jong Wook Kim; Xu, Tao; Brockman, Greg; McLeavey, Christine; Sutskever, Ilya (2022). "Robust Speech Recognition via Large-Scale Weak Supervision". arXiv:2212.04356 [eess.AS].
    \bibitem{monastirsky2023} Monastirsky, Maxim; Azulay, Osher; Sintov, Avishai (February 2023). "Learning to Throw With a Handful of Samples Using Decision Transformers". IEEE Robotics and Automation Letters. 8 (2): 576\u2013583.
    \bibitem{wolf2020} Wolf, Thomas; Debut, Lysandre; Sanh, Victor; Chaumond, Julien; Delangue, Clement; Moi, Anthony; Cistac, Pierric; Rault, Tim; Louf, Remi; Funtowicz, Morgan; Davison, Joe; Shleifer, Sam; von Platen, Patrick; Ma, Clara; Jernite, Yacine; Plu, Julien; Xu, Canwen; Le Scao, Teven; Gugger, Sylvain; Drame, Mariama; Lhoest, Quentin; Rush, Alexander (2020). "Transformers: State-of-the-Art Natural Language Processing". Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. pp.\u00a038\u201345.
    \end{thebibliography}
    \end{document}
    ''')

    stages = [
        ("Stage 1: Irrelevant Content", stage1_content),
        ("Stage 2: Relevant Content, Poor Plan", stage2_content),
        ("Stage 3: Relevant Content, Good Plan", stage3_content),
        ("Stage 4: Full (Content + Plan + Few Bib)", stage4_content),
        ("Stage 5: Full (Content + Plan + Many Bib)", stage5_content),
    ]

    # Pre-compute golden entities to speed up evaluation
    from common.metrics import get_entities_from_article
    from common.doc_eval import temporary_transform_dea_into_markdown
    
    print("Pre-computing golden entities (this runs NER once)...")
    reference_content = temporary_transform_dea_into_markdown(solution)
    golden_entities = get_entities_from_article(reference_content)
    print(f"Found {len(golden_entities)} entities in reference.")

    import os
    backends = ["hf"]
    if os.getenv("OPENAI_API_KEY"):
        backends.append("openai")

    for backend in backends:
        print(f"\n\n{'='*40}")
        print(f"Running Demo with Backend: {backend}")
        print(f"{'='*40}")

        for name, content in stages:
            # Only run Stage 4 and 5 for OpenAI to save time/tokens, unless user wants full run
            if backend == "openai" and "Stage 4" not in name and "Stage 5" not in name:
                continue
                
            print(f"\nRunning {name} ({backend})...")
            try:
                scores = evaluate_document(
                    content,
                    solution=solution,
                    content_type="latex",
                    skip_dea=False,
                    use_enhanced_metrics=USE_ENHANCED_METRICS,
                    dea_embedding_backend=backend,
                    openai_model="gpt-4o" if os.getenv("OPENAI_API_KEY") else None,
                    golden_entities=golden_entities
                )
                print_scores(name, scores)
            except Exception as e:
                print(f"Error running {backend}: {e}")

if __name__ == "__main__":
    main()
