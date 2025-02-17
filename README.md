# 📄 Document Embedding Analysis
*Harnessing embeddings for direct content comparison analysis.*

## 🎤 Introduction

Given an outline of an article (all the headings and subheadings), how well can a Large Language Model (LLM) generate a similar unknown content? That's the purpose of this dataset. 

Taking an Arxiv paper, a patent, or a Wikipedia article like [Wikipedia page for self-driving cars](https://en.wikipedia.org/wiki/Self-driving_car) as an example, we see the following headings: **Definitions, Automated driver assistance system, Autonomous vs. automated, Autonomous versus cooperative** and associated content for each heading, etc. We then try to generate a similar plan and content without prior knowledge of this content (only title and abstract), and compare this content with the ground truth.

We built this dataset to be able to compare semantic structure and content for each section, as well as its length. Then, for any sections that were longer than a max embedding token limit, split section up and give them new unique titles. 

Rverything extracted (headings and content) is then analysed - e.g., calculating the Rouge-L, MAUVE and cosine similarity scores. Some scores would only accept a max number of tokens (we had to split them above).

## 💻 How to Run the Code

#### 1. Download code + create env
```
$ git clone https://github.com/doxav/document_extraction.git
$ cd document_extraction
$ pip install -r requirements.txt
```

#### 2. Set your OpenAI key (it supports both Huggingface Models and OpenAI)

1. Go to https://platform.openai.com/account/api-keys to create a new key (if you don't have one already).
2. Rename `.env_template` to `.env`
3. Add your key in next to `OPENAI_API_KEY`
