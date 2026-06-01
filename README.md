# BLaIR Dual-Encoder Retrieval

High-performance dense retrieval on Amazon product corpus.

## Exploratory Data Analysis
Token length distributions analyzed across product reviews.
Product categories analyzed: Electronics, Clothing, Home.

### BM25 Baseline
BM25 achieved NDCG@10 = 0.0208 on product search.
| Model | NDCG@10 | Recall@10 |
|---|---|---|
| BM25 | 0.0208 | 0.0410 |

### BiEncoder Results
Fine-tuned BiEncoder achieved NDCG@10 = 0.0812.

## Training Hyperparameters
Batch size 64, LR 2e-5, Cosine scheduler, tau 0.05.
