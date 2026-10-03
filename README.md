# 🔎 Stripe Help Center Search Engine

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas&logoColor=white)
![BM25](https://img.shields.io/badge/Retrieval-BM25-orange?style=flat)
![Word2Vec](https://img.shields.io/badge/Embeddings-Word2Vec-8E44AD?style=flat)
![Search](https://img.shields.io/badge/Task-Information%20Retrieval-blue?style=flat)
![Status](https://img.shields.io/badge/Status-Prototype-yellow?style=flat)

> **Build a search engine that understands customer questions written in their own words, ranks the most relevant Stripe Help Center articles, evaluates retrieval quality, and produces the top three article recommendations for held-out questions.**

---

## 📌 Project Overview

Stripe's Help Center contains articles covering a wide range of customer questions. However, customers do not necessarily use the same terminology as the documentation.

For example, an article may describe a process using formal product terminology while a customer may ask:

> *"money didn't reach my bank"*

A search system based only on exact word matching can struggle when the customer's wording differs from the wording used in the article.

This project explores an **information retrieval system** that combines lexical matching with semantic similarity to improve article ranking.

The notebook currently builds a hybrid retrieval prototype using:

- **BM25** for lexical relevance
- **Word2Vec** for semantic similarity
- **Cosine similarity** for comparing query and article vectors
- **Min-Max normalization** to place the two scoring methods on a comparable scale
- **Weighted score fusion** to produce a final ranking

The project is based on three datasets:

| Dataset | Rows | Purpose |
|---|---:|---|
| `articles.csv` | 24 | Stripe Help Center articles |
| `queries_train.csv` | 216 | Training questions with known relevant articles |
| `queries_test.csv` | 96 | Held-out questions requiring predictions |

---

# 🎯 Problem Statement

Build a search system that:

1. Reads the Help Center articles and training questions.
2. Identifies differences between customer language and article language.
3. Creates a simple lexical retrieval baseline.
4. Measures retrieval quality using:
   - **Recall@3**
   - **Mean Reciprocal Rank (MRR)**
5. Improves retrieval using methods such as BM25, semantic similarity, title weighting, stop-word handling, or training-question language.
6. Avoids evaluation leakage when training questions are used to enhance the index.
7. Investigates questions that remain incorrectly ranked.
8. Returns the **top three articles** for every test question.
9. Saves the final predictions in the required format.

---

# 📂 Dataset

## 1. `articles.csv`

Contains the 24 Stripe Help Center articles.

| Column | Description |
|---|---|
| `article_id` | Unique article identifier |
| `category` | Help Center section |
| `title` | Article title |
| `body` | Article content |

### Example article fields

An article contains information such as:

```text
article_id
category
title
body
```

The notebook reconstructs the article collection from the training data by selecting:

```python
["relevant_article_id", "category", "title", "body"]
```

and removing duplicates.

This produces the **24 unique Help Center articles** used as the retrieval corpus.

---

## 2. `queries_train.csv`

Contains 216 customer questions and the article that answers each question.

| Column | Description |
|---|---|
| `query_id` | Unique question identifier |
| `query` | Customer's natural-language question |
| `relevant_article_id` | Correct article for the question |

Example questions in the notebook include:

| Query | Relevant Article | Topic |
|---|---|---|
| `why set trial lentgh to 14 days asap` | `A121` | Free trials |
| `how do i fight a chargeback?` | `A105` | Responding to a dispute |
| `how do i upload id to keep my account active?` | `A116` | Business identity verification |
| `hi, dunning settings for subscriptions` | `A122` | Failed subscription payments |
| `need help: upgrade a subscriber mid month on m...` | `A120` | Subscription plan changes |

These examples demonstrate that customer questions may be:

- Informal
- Abbreviated
- Typo-containing
- Written using product-specific language
- Different from the wording in the article title

---

## 3. `queries_test.csv`

Contains 96 held-out customer questions.

| Column | Description |
|---|---|
| `query_id` | Unique question identifier |
| `query` | Customer question |

The test questions do not contain the ground-truth article ID, so the search system must rank the articles and return the top three predictions.

---

# 🔍 Information Retrieval Challenge

The central challenge is the **vocabulary mismatch** between customers and documentation.

A customer may describe an action or problem in everyday language while the Help Center uses Stripe's formal terminology.

For example:

```text
Customer language
        ↓
"fight a chargeback"

Article language
        ↓
"Responding to a dispute"
```

Another example from the training data:

```text
Customer:
"upload id to keep my account active"

Article:
"Verifying your business identity"
```

This means a useful search engine should not rely exclusively on exact word overlap.

---

# 🧹 Data Preparation

The notebook performs the following preparation steps.

### 1. Load the training data

```python
df = pd.read_csv("queries_train.csv")
```

### 2. Extract the unique article corpus

```python
articles_df = (
    df[["relevant_article_id", "category", "title", "body"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
```

The training dataset contains repeated references to the same articles, so duplicates are removed to create the 24-document article corpus.

### 3. Combine article information

The notebook combines the article ID, category, title, and body:

```python
articles_df["combined_text"] = (
    articles_df["relevant_article_id"]
    + " "
    + articles_df["category"]
    + " "
    + articles_df["title"]
    + " "
    + articles_df["body"]
)
```

### 4. Tokenize the corpus

The current prototype uses simple lowercase whitespace tokenization:

```python
tokenized_corpus = [
    doc.lower().split()
    for doc in articles_df["combined_text"]
]
```

---

# 🏗️ Search Architecture

The current prototype uses a **hybrid retrieval architecture**.

```text
                    Customer Query
                           │
                           ▼
                 Lowercase + Tokenize
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
          BM25 Search              Word2Vec
              │                  Query Embedding
              │                         │
              ▼                         ▼
        BM25 Scores             Article Embeddings
              │                         │
              └────────────┬────────────┘
                           ▼
                  Min-Max Normalization
                           │
                           ▼
                    Score Combination
                           │
                           ▼
                    Ranked Articles
                           │
                           ▼
                       Top 3
```

---

# 🥇 Retrieval Method 1 — BM25

The notebook uses:

```python
from rank_bm25 import BM25Okapi
```

The BM25 index is created from the 24 article documents:

```python
bm25 = BM25Okapi(tokenized_corpus)
```

For a query:

```python
query_token = query_text.lower().split()

bm25_scores = bm25.get_scores(query_token)
```

### Why BM25?

BM25 is a strong classical information-retrieval method because it considers:

- Term frequency
- Document frequency
- Document length
- How informative a term is across the corpus

Compared with simple word-count matching, BM25 provides a more effective lexical ranking mechanism.

---

# 🧠 Retrieval Method 2 — Word2Vec

The notebook also trains a Word2Vec model:

```python
w2vec = Word2Vec(
    tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=5
)
```

Each article is converted into a single vector by averaging the Word2Vec vectors of the words it contains.

```python
def get_document_vector(tokenized_text, model):
    vectors = [
        model.wv[word]
        for word in tokenized_text
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)
```

This creates one vector representation per article.

---

# 🔗 Hybrid Retrieval

The prototype combines BM25 and Word2Vec similarity.

### Step 1 — BM25 score

```python
bm25_scores = bm25.get_scores(query_token)
```

### Step 2 — Query embedding

```python
query_vector = get_document_vector(query_token, w2vec)
```

### Step 3 — Cosine similarity

```python
w2v_scores = cosine_similarity(
    [query_vector],
    article_vectors
).flatten()
```

### Step 4 — Normalize scores

Both score arrays are normalized using `MinMaxScaler`.

```python
bm25_norm = scaler.fit_transform(
    bm25_scores.reshape(-1, 1)
).flatten()

w2v_norm = scaler.fit_transform(
    w2v_scores.reshape(-1, 1)
).flatten()
```

### Step 5 — Combine scores

The final hybrid score is:

```python
hybrid_scores = (
    alpha * bm25_norm
    + (1 - alpha) * w2v_norm
)
```

With the current default:

```python
alpha = 0.5
```

the final score gives equal weight to:

- **50% BM25**
- **50% Word2Vec similarity**

The articles are then sorted by the hybrid score and the top three are returned.

---

# 📏 Evaluation Metrics

The task requires two standard information-retrieval metrics.

## Recall@3

Recall@3 measures whether the correct article appears anywhere in the top three results.

```text
Recall@3 =
Number of queries where correct article is in top 3
----------------------------------------------------
                 Total queries
```

For example, if 180 of 216 questions have the correct article in the top three:

```text
Recall@3 = 180 / 216
         = 0.8333
         = 83.33%
```

---

## Mean Reciprocal Rank — MRR

MRR also considers **where** the correct article appears.

| Correct Article Position | Reciprocal Rank |
|---:|---:|
| 1 | 1.00 |
| 2 | 0.50 |
| 3 | 0.33 |
| Not in top 3 | 0.00 |

Formula:

```text
MRR = Average(1 / rank)
```

A system that consistently puts the correct article first will therefore have a higher MRR than a system that usually places it second or third.

---

# ⚠️ Evaluation Status of the Current Notebook

The uploaded notebook contains the hybrid search implementation and generates a test prediction file, but the evaluation section is **not currently executable end-to-end**.

The `evaluate_search()` function expects fields named:

```python
customer_query
true_article_code
```

while the training data actually uses:

```python
query
relevant_article_id
```

The function also expects search results containing:

```python
article_code
```

while `hybrid_search()` currently constructs results using:

```python
article_id
```

In addition, the `hybrid_search()` result-building loop references:

```python
row
```

without defining `row` inside that function.

Therefore, **no valid Recall@3 or MRR values are claimed in this README from the current notebook**.

This is intentionally documented rather than inventing evaluation results.

---

# 🧪 Evaluation Leakage Consideration

The project instructions require an honest evaluation when training questions are used to improve retrieval.

The current notebook builds its retrieval corpus from the unique article content only:

```text
24 Help Center articles
```

It does **not** currently append customer questions to the article documents.

That avoids one form of query-to-document leakage.

However, if future versions enrich each article with the wording of its training questions, the evaluation must use cross-validation.

For example:

```text
Training fold
     ↓
Learn customer phrasing
     ↓
Build enriched article representations
     ↓
Evaluate on held-out questions
```

A query must never contribute its own wording to the representation used to retrieve itself.

---

# 🧪 Recommended Search Versions

The project is designed to compare retrieval approaches progressively.

## Version 1 — Simple Word Overlap Baseline

A baseline can rank articles according to the number or proportion of words shared with the query.

```text
Query
  ↓
Tokenize
  ↓
Count shared terms
  ↓
Rank articles
```

### Purpose

This establishes how well a very simple lexical system performs before introducing stronger retrieval techniques.

---

## Version 2 — BM25

Replace raw word overlap with BM25.

```text
Query
  ↓
BM25
  ↓
Rank articles
```

BM25 should provide a stronger lexical baseline by accounting for term frequency, document frequency, and document length.

---

## Version 3 — BM25 + Word2Vec

The current notebook implements this hybrid approach:

```text
                ┌── BM25 ───────────┐
Query ──────────┤                   ├── Weighted Score ──► Ranking
                └── Word2Vec ───────┘
```

Current weighting:

```text
BM25      = 50%
Word2Vec  = 50%
```

---

## Version 4 — Training-Question Enrichment

A further improvement would be to attach customer wording to the corresponding article.

For example:

```text
Article:
"Responding to a dispute"

Training questions:
"how do i fight a chargeback?"
"customer says they didn't make the payment"
"how can I challenge a dispute?"
```

This can bridge the vocabulary gap between customer language and article terminology.

However, this version should be evaluated using **cross-validation** to avoid using the evaluation query itself during retrieval-model construction.

---

# 📊 Model Comparison

The final README should compare every implemented retrieval version using the same evaluation dataset.

| Search Version | Method | Recall@3 | MRR |
|---|---|---:|---:|
| Version 1 | Word-overlap baseline | Not calculated | Not calculated |
| Version 2 | BM25 | Not calculated | Not calculated |
| Version 3 | BM25 + Word2Vec | Not calculated | Not calculated |
| Version 4 | Training-question enrichment + CV | Not implemented | Not implemented |

> **Note:** The current notebook does not contain valid metric outputs for these versions, so the table deliberately does not fabricate numbers.

---

# 🧪 Test Prediction Pipeline

The notebook loads the held-out test questions:

```python
test_df = pd.read_csv("queries_test.csv")
```

For each query, it requests the top three articles:

```python
top_3 = hybrid_search(
    q_text,
    top_k=3
)
```

The intended output contains:

```text
query_id
rank_1
rank_2
rank_3
```

The current notebook additionally stores the query and article titles for inspection.

---

# 📄 Prediction Output

The task requires:

```text
predictions.csv
```

with exactly these columns:

| Column | Description |
|---|---|
| `query_id` | Test query identifier |
| `rank_1` | Highest-ranked article |
| `rank_2` | Second-ranked article |
| `rank_3` | Third-ranked article |

### Example

```csv
query_id,rank_1,rank_2,rank_3
Q5001,A117,A112,A105
Q5002,A117,A104,A108
Q5003,A114,A112,A106
```

The uploaded notebook currently writes:

```text
test_predictions.csv
```

and includes additional columns such as:

```text
query
rank_1_title
rank_2_title
rank_3_title
```

The final submission should be transformed to the exact required schema:

```text
query_id,rank_1,rank_2,rank_3
```

---

# 🔎 Sample Test Predictions from the Notebook

The notebook successfully generates a test prediction table. The first rows shown in the notebook include:

| Query ID | Query | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|---|
| `Q5001` | help extra security code at sign in | `A117` | `A112` | `A105` |
| `Q5002` | fee for instant transfer | `A117` | `A104` | `A108` |
| `Q5003` | why take payment after shipping | `A114` | `A112` | `A106` |
| `Q5004` | refund bounced because card was cancelled | `A111` | `A101` | `A119` |
| `Q5005` | why switch pricing tier for a user on my account | `A117` | `A114` | `A101` |

These are the notebook's generated rankings and are **not presented as accuracy-verified predictions**, because the test set does not contain the ground-truth article IDs.

---

# ❌ Failure Analysis

The project requires identifying three questions that the search system still gets wrong.

The current notebook does **not** implement a ground-truth comparison for the test dataset, because `queries_test.csv` does not contain `relevant_article_id`.

Therefore, a defensible three-error analysis cannot be produced from the current notebook without either:

1. Evaluating the training questions correctly, or
2. Having ground-truth labels for the test questions.

A future failure-analysis section should contain:

| Query | Expected Article | Predicted Article | Why It Failed |
|---|---|---|---|
| Query 1 | Article ID | Article ID | Vocabulary mismatch / semantic ambiguity |
| Query 2 | Article ID | Article ID | Weak title signal / overlapping terminology |
| Query 3 | Article ID | Article ID | Missing phrase / insufficient semantic representation |

This prevents unsupported error explanations from being presented as facts.

---

# 🛠️ Technical Issues Identified in the Current Notebook

The notebook provides a useful prototype, but several implementation issues should be corrected before treating it as a completed submission.

### 1. Undefined `row` in `hybrid_search()`

The function creates:

```python
for idx in top_indices:
    results.append({
        "article_id": row["relevant_article_id"],
        ...
    })
```

but `row` is never defined.

It should instead retrieve the article using the selected `idx`.

---

### 2. Inconsistent article-ID field names

The search function creates:

```python
"article_id"
```

but the evaluation and prediction code expects:

```python
"article_code"
```

These should be standardized to:

```text
article_id
```

---

### 3. Inconsistent query field names

The evaluation function expects:

```python
customer_query
```

while the actual training data contains:

```python
query
```

The evaluator should use:

```python
item["query"]
```

---

### 4. Inconsistent ground-truth field names

The evaluation function expects:

```python
true_article_code
```

while the dataset contains:

```python
relevant_article_id
```

The evaluator should use:

```python
item["relevant_article_id"]
```

---

### 5. Evaluation is not actually executed

Although `evaluate_search()` is defined, the notebook does not provide a valid call that evaluates the training questions and reports Recall@3 and MRR.

---

### 6. Required output filename/schema differs

The task requests:

```text
predictions.csv
```

with:

```text
query_id
rank_1
rank_2
rank_3
```

The notebook currently creates:

```text
test_predictions.csv
```

with additional columns.

---

### 7. No true baseline implementation

The instructions ask for a simple shared-word baseline first.

The notebook currently jumps to:

```text
BM25 + Word2Vec
```

A true baseline should be implemented and evaluated before the improved search system.

---

# 🚀 Recommended Final Project Workflow

A clean final implementation should follow this sequence:

```text
                ┌─────────────────────┐
                │  Load 3 datasets    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Inspect vocabulary  │
                │ & customer wording  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Word-overlap        │
                │ baseline            │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Recall@3 + MRR      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ BM25                │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Recall@3 + MRR      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ BM25 + Word2Vec     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Recall@3 + MRR      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Optional query      │
                │ enrichment + CV     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Analyze failures    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Rank 96 test        │
                │ questions           │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ predictions.csv     │
                └─────────────────────┘
```

---

# 📦 Libraries

The notebook uses the following Python libraries:

```python
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
```

Required external packages include:

```bash
pip install pandas numpy gensim rank_bm25 scikit-learn
```

---

# 📁 Suggested Repository Structure

```text
stripe-help-center-search/
│
├── 📁 data/
│   ├── articles.csv
│   ├── queries_train.csv
│   └── queries_test.csv
│
├── 📁 outputs/
│   └── predictions.csv
│
├── 📁 notebooks/
│   └── Stripe_Help_Center_Search.ipynb
│
├── 📄 README.md
└── 📄 requirements.txt
```

---

# 🎯 Business Value

A better Help Center search system can reduce the effort required for customers to find answers.

Potential benefits include:

- 🔎 Faster article discovery
- 📉 Fewer failed searches
- 💬 Better self-service support
- ⏱️ Reduced time spent searching
- 📞 Potential reduction in support escalation
- 📚 Better understanding of customer terminology

The most important technical challenge is not simply retrieving documents, but **connecting the language used by customers with the language used by documentation**.

---

# 🔮 Future Improvements

Several improvements can make the system more robust.

### 1. TF-IDF Baseline

Add a TF-IDF + cosine similarity model as another classical retrieval benchmark.

### 2. Better Tokenization

Improve tokenization by handling:

- Punctuation
- Typos
- Contractions
- Product-specific terms
- Hyphenated words

### 3. Stop-Word Removal

Evaluate whether removing common words improves retrieval quality.

### 4. Title Weighting

Give article titles greater importance than body text because titles often provide a concise description of the article topic.

For example:

```text
Score =
0.7 × title_score
+
0.3 × body_score
```

The exact weights should be selected using validation rather than assumed.

### 5. Query Expansion

Use training questions to learn alternate customer expressions for each article.

Example:

```text
Article:
Responding to a dispute

Customer vocabulary:
chargeback
fight chargeback
challenge payment
dispute payment
customer disputed charge
```

### 6. Cross-Validation

When training questions are incorporated into the retrieval index, use cross-validation to prevent a query from retrieving itself because its own wording was added to the article representation.

### 7. Better Semantic Embeddings

The current Word2Vec model is trained on only the small article corpus. A stronger future approach could use a pretrained sentence/document embedding model, while still respecting the project requirement to avoid large language models or paid APIs if those constraints apply.

---

# 📌 Conclusion

This project explores how classical information-retrieval and lightweight semantic techniques can be combined to improve Help Center search.

The current notebook establishes a prototype based on:

```text
BM25
  +
Word2Vec
  +
Cosine Similarity
  +
Min-Max Normalization
  +
Weighted Score Fusion
```

The most important design principle is to evaluate each retrieval version consistently using **Recall@3 and MRR**, while avoiding leakage when customer questions are incorporated into the search index.

The uploaded notebook successfully demonstrates the core hybrid-ranking idea and produces top-three test rankings, but it still requires a few implementation corrections before it can be considered a complete, metrics-backed submission.

---

## ⭐ Project Highlights

- 📚 24 Help Center articles
- 🔎 216 labeled training questions
- 🧪 96 held-out test questions
- 🥇 BM25 lexical retrieval
- 🧠 Word2Vec semantic representation
- 🔗 Hybrid ranking
- 📏 Recall@3 evaluation framework
- 📊 MRR evaluation framework
- 🚫 Leakage-aware evaluation design
- 📄 Top-3 prediction generation
- 🔬 Failure-analysis framework

---

**Built with Python 🐍 | BM25 🔎 | Word2Vec 🧠 | Scikit-learn 📊 | Information Retrieval 🚀**
