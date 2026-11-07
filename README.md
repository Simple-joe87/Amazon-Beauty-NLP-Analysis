# Amazon Beauty Review Complaint Topic Modelling (IU Data Analysis)

## Objective
Discover recurring dissatisfaction themes in 1–2-star reviews from McAuley Lab's Amazon Reviews 2023 `All_Beauty` category.

## Data
Download `All_Beauty.jsonl` from https://amazon-reviews-2023.github.io/ and place it in the project directory. Do not commit the dataset or generated review text to a public repository.

## Reproduce
```bash
python -m venv .venv
# Activate the environment for your operating system
pip install -r requirements.txt
python analyze.py --input All_Beauty.jsonl --output results --sample 10000
```

The script filters 1–2-star reviews, requires >=10 whitespace-delimited words, deduplicates case-insensitive whitespace-normalized text, and samples with random seed 42. It creates CountVectorizer and TF-IDF matrices, fits 8-topic LDA and NMF models, and writes results/summary.json, results/topics.csv, results/sample_metadata.csv. `topics.csv` and `sample_metadata.csv` include review excerpts: do not publish these without checking dataset terms and privacy implications.

## Preliminary findings
The 10,000-review pilot produces some interpretable themes (hair products, appearance mismatch, nonfunctioning products, perceived wasted money, packaging/arrival issues), but some topics are generic or contaminated by HTML `br` tokens. Topic counts are dominant-topic assignments, not representative prevalence across all customers. Further cleaning, topic-number sensitivity tests, and manual document validation are needed before the final analysis.

## Methods and citations
- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging language and items for retrieval and recommendation*. https://doi.org/10.48550/arXiv.2403.03952
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022. https://www.jmlr.org/papers/v3/blei03a.html
- Scikit-learn developers. (n.d.). *Topic extraction with non-negative matrix factorization and latent Dirichlet allocation*. https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
