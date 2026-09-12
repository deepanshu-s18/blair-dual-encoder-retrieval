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

### Scaling Analysis
DualEncoder outperforms BiEncoder at scale >= 50k samples.

### Asymmetric DualEncoder
Separate query and document backbones.

### Hybrid Retrieval
Hybrid RRF achieves NDCG@10 = 0.0988.
| Model | NDCG@10 |
|---|---|
| Hybrid (RRF) | 0.0988 |

### Latency vs Accuracy Pareto
Cross-encoder brings +4% NDCG at 28x latency cost.

### Embedding Geometry
t-SNE reveals 70.2% tighter clustering after contrastive training.

## Error Taxonomy
Five distinct failure categories analyzed.

### Modern Baselines
BGE zero-shot outperforms initial BiEncoder.
| BGE (Fine-tuned) | **0.0995** |

### Business Impact
+10% NDCG translates to estimated +2.8% Conversion Rate.

### Amazon DSA Prep
Binary search, two-pointer, and sliding window problems implemented.
Graph algorithms: BFS, DFS, Dijkstra, Topological Sort.
