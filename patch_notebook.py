"""
patch_notebook.py
=================
Injects the 4 missing sections into Amazon_ML_Interview_Complete.ipynb:
  1. KNN full section (2.11) — appeared in real 2025 Amazon interviews
  2. SVM + kernel trick (2.12) — in ToC but missing from body
  3. Cross-validation (2.13) — in Amazon companion, not in body or ToC
  4. Updates Table of Contents with Why Amazon + missing sections

Also updates the ToC cell (Cell 00) to include cross-validation and Why Amazon.
Run: python3 patch_notebook.py
"""

import json
import copy
from pathlib import Path

NB_PATH = "Amazon_ML_Interview_Complete.ipynb"

def md_cell(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

def code_cell(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }


# ── New Section: KNN ─────────────────────────────────────────────────────────
KNN_MD = md_cell("""## 2.11 KNN — K Nearest Neighbors (Full Theory)

**[REAL Q]** Appeared in real 2025 Amazon interviews + Amazon companion document.

### WHAT:
Find k nearest training points to a test point, return majority vote (classification) or mean (regression).

### Time Complexity:
| Phase | Complexity |
|-------|-----------|
| Training | O(1) — just store data |
| Prediction | O(N×d) — distance to all N training points |

At N=56,921, d=768: ~44M multiplications per query. **This is WHY FAISS exists.**

### Choosing k:
```
k = 1  → high variance (overfits to noise — one bad neighbor decides everything)
k = N  → high bias (always predicts majority class)
k = √N → rule of thumb starting point; cross-validate for final choice
```

### [REAL Q] "What's the connection to your FAISS retrieval system?"
> "FAISS IndexFlatIP IS exact KNN on 56,921 product embeddings. For each query, it computes cosine similarity to all N products and returns top-k — that's K Nearest Neighbors. The difference: we use INNER PRODUCT (cosine similarity for L2-normalized vectors) not Euclidean distance. At 500M products, exact KNN is O(500M × 768) per query = too slow, so we switch to approximate KNN via IVFFlat or HNSW."

### [TRAP Q] "Why Euclidean distance vs cosine similarity?"
```
Euclidean: ||a-b||₂  — sensitive to magnitude (long docs score higher just because they're longer)
Cosine:    a·b / (||a||×||b||)  — magnitude-invariant, only angle matters

For text embeddings: ALWAYS use cosine similarity (or dot product after L2 norm).
After L2 normalization: cosine similarity = dot product (cheaper to compute).
```

### Curse of Dimensionality:
In high dimensions (d=768), ALL points are nearly equidistant from any query:
- Ratio of max/min distance → 1 as d → ∞
- KNN neighbors stop being "near" — they're all equally far
- Fix: dimensionality reduction (PCA → 50d) before KNN, or use learned embeddings
""")

KNN_CODE = code_cell("""import numpy as np
import matplotlib.pyplot as plt

def knn_predict(X_train, y_train, x_new, k=5):
    \"\"\"
    Full KNN from scratch.
    Connection to BLaIR: FAISS IndexFlatIP = exact KNN with inner product metric.
    At 56,921 products: O(N×d) is fast. At 500M: use IVFFlat (ANN).
    \"\"\"
    # Euclidean distance from x_new to ALL training points: O(N×d)
    distances = np.array([np.linalg.norm(x - x_new) for x in X_train])
    
    # Sort ascending, take k smallest
    k_indices = np.argsort(distances)[:k]
    k_labels  = y_train[k_indices].tolist()
    
    # Majority vote
    return int(max(set(k_labels), key=k_labels.count))


def knn_predict_cosine(embs, corpus_ids, query_emb, k=10):
    \"\"\"
    KNN with COSINE similarity — this is what FAISS IndexFlatIP does.
    For L2-normalized vectors: cosine = dot product.
    \"\"\"
    # L2-normalize
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    embs_norm  = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    
    # Cosine similarities (= dot products after normalization): O(N×d)
    scores = embs_norm @ query_emb   # (N,)
    
    # Top-k indices (descending)
    k_indices = np.argsort(scores)[::-1][:k]
    return [(corpus_ids[i], float(scores[i])) for i in k_indices]


# ── DEMO: KNN on 2D data ────────────────────────────────────────────────────
np.random.seed(42)

# 3 clusters of training data
X_class0 = np.random.randn(30, 2) * 0.5 + [0, 0]
X_class1 = np.random.randn(30, 2) * 0.5 + [3, 0]
X_class2 = np.random.randn(30, 2) * 0.5 + [1.5, 2.5]
X_train = np.vstack([X_class0, X_class1, X_class2])
y_train = np.array([0]*30 + [1]*30 + [2]*30)

# Test point
x_test = np.array([1.6, 2.3])
pred_k1 = knn_predict(X_train, y_train, x_test, k=1)
pred_k5 = knn_predict(X_train, y_train, x_test, k=5)
pred_k15 = knn_predict(X_train, y_train, x_test, k=15)

# Decision boundary visualization (grid)
xx, yy = np.meshgrid(np.linspace(-2, 5, 80), np.linspace(-1, 4, 80))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([knn_predict(X_train, y_train, pt, k=5) for pt in grid]).reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: scatter + prediction
colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71'}
for c in [0, 1, 2]:
    ax1.scatter(X_train[y_train==c, 0], X_train[y_train==c, 1],
                c=colors[c], alpha=0.6, s=30, label=f'Class {c}')
ax1.scatter(*x_test, c='black', s=200, marker='★', zorder=10, label='Query point')
ax1.set_title(f'KNN Predictions for Query ★\\nk=1: class {pred_k1} | k=5: class {pred_k5} | k=15: class {pred_k15}')
ax1.legend(fontsize=8)

# Right: decision boundary (k=5)
ax2.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5],
             colors=['#e74c3c', '#3498db', '#2ecc71'])
for c in [0, 1, 2]:
    ax2.scatter(X_train[y_train==c, 0], X_train[y_train==c, 1],
                c=colors[c], alpha=0.7, s=20)
ax2.set_title('KNN Decision Boundary (k=5)\\nNote: nonlinear boundary — no training needed')

plt.tight_layout()
plt.show()

print(f"k=1  prediction: class {pred_k1}  (overfits nearest neighbor)")
print(f"k=5  prediction: class {pred_k5}  (good tradeoff)")
print(f"k=15 prediction: class {pred_k15}  (smoother — may underfit)")
print()
print("FAISS connection:")
print("  DenseRetriever.retrieve(query_emb, k=10) = exact KNN with cosine similarity")
print("  N=56,921 products, d=768: O(N×d) ≈ 44M ops per query (fast with SIMD)")
print("  N=500M products: use IVFFlat (nprobe=64 → search 6.25% of corpus)")
""")

# ── New Section: SVM + Kernel Trick ──────────────────────────────────────────
SVM_MD = md_cell("""## 2.12 SVM — Support Vector Machines + Kernel Trick

**[REAL Q]** In ToC as 2.12. Amazon companion lists SVM + kernel trick as high-priority.

### WHAT:
Find the hyperplane that separates classes with **maximum margin**.
Margin = distance between hyperplane and nearest points (support vectors).

### WHY maximum margin:
The hyperplane with the largest margin has the best generalization — it's furthest from all training points, so small perturbations in new data don't cross the boundary.

### Hard-Margin SVM (linearly separable data):
```
Objective:     minimize  ||w||²/2
Constraints:   yᵢ(w·xᵢ + b) ≥ 1  for all i

Decision boundary:  w·x + b = 0
Support vectors:    w·x + b = ±1  (closest points to boundary)
Margin width:       2 / ||w||    (maximized by minimizing ||w||²)
```

### Soft-Margin SVM (real data — not linearly separable):
```
minimize:  ||w||²/2 + C × Σᵢ ξᵢ
constraint: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0

ξᵢ = slack variable (how much sample i violates the margin)
C  = regularization: small C = wide margin (more violations), large C = hard margin
```

### [REAL Q] THE KERNEL TRICK:
What if data is NOT linearly separable in input space?

**Key insight:** SVM only needs DOT PRODUCTS between points, not coordinates.

```python
# Linear SVM uses: x_i · x_j
# Kernel SVM uses: K(x_i, x_j) = φ(x_i) · φ(x_j)
# where φ is a feature map to a higher-dimensional space
# TRICK: we never compute φ(x) explicitly — just K(x_i, x_j) directly!
```

| Kernel | Formula | Separates |\n|--------|---------|----------|\n| Linear | x_i · x_j | Linear boundaries |\n| RBF/Gaussian | exp(-γ‖xᵢ-xⱼ‖²) | Circular/complex boundaries |\n| Polynomial | (xᵢ·xⱼ + c)^d | Polynomial curves |\n| Sigmoid | tanh(α xᵢ·xⱼ + c) | Similar to 1-layer NN |

### WHY RBF kernel is default:
RBF (Radial Basis Function) maps data to infinite-dimensional space. It can separate ANY distribution if γ and C are tuned correctly.

### [REAL Q] "SVM vs Logistic Regression — when to use which?"
```
SVM:
  ✓ High-dimensional sparse features (text, gene expression)
  ✓ Small datasets (kernel trick is O(n²) or O(n³))
  ✓ Clear margin exists
  ✗ Slow for large N (doesn't scale)
  ✗ No probability estimates without calibration

Logistic Regression:
  ✓ Large datasets
  ✓ Need probability outputs
  ✓ Interpretable (log-odds = linear combination of features)
  ✗ Only linear boundary (unless feature engineering)
```

### Connection to YOUR project:
> "SVM would be catastrophically slow on our problem — for 56,921 products and 79,703 training pairs, kernel matrix is 79,703² ≈ 6.4 billion entries. Our FAISS-based KNN with learned embeddings is O(N×d) per query, far more scalable. However, the margin-maximization intuition from SVM is related to InfoNCE: both push representations of different classes apart."
""")

SVM_CODE = code_cell("""import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ── Visualize the Kernel Trick ──────────────────────────────────────────────
# 1D data that is NOT linearly separable → 2D feature map makes it separable

np.random.seed(42)

# 1D: two "rings" around origin — NOT linearly separable
x_neg = np.concatenate([
    np.random.uniform(-4, -1.5, 20),
    np.random.uniform(1.5, 4, 20)
])
x_pos = np.random.uniform(-1, 1, 20)

X_1d = np.concatenate([x_neg, x_pos])
y    = np.array([-1]*40 + [1]*20)

# Kernel trick: map x → (x, x²) — polynomial degree 2 in 2D
def phi(x):
    \"\"\"Feature map: x → [x, x²]  (implicit in kernel trick)\"\"\"
    return np.column_stack([x, x**2])

X_2d = phi(X_1d)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: 1D data — NOT separable
ax = axes[0]
ax.scatter(x_neg, np.zeros_like(x_neg), c='#e74c3c', s=50, label='Class -1', zorder=5)
ax.scatter(x_pos, np.zeros_like(x_pos), c='#3498db', s=50, label='Class +1', zorder=5)
ax.set_ylim(-0.5, 0.5); ax.set_xlabel('x')
ax.set_title('1D Input Space\\nNOT linearly separable', fontsize=10)
ax.legend(fontsize=8)
ax.axhline(0, color='gray', alpha=0.3)

# Panel 2: 2D feature space — SEPARABLE (kernel trick)
ax = axes[1]
ax.scatter(X_2d[y==-1, 0], X_2d[y==-1, 1], c='#e74c3c', s=50, label='Class -1', zorder=5)
ax.scatter(X_2d[y== 1, 0], X_2d[y== 1, 1], c='#3498db', s=50, label='Class +1', zorder=5)
# Draw separating line in 2D space
xlim = ax.get_xlim() if ax.get_xlim() != (0,1) else (-4.5, 4.5)
x_line = np.linspace(-4.5, 4.5, 100)
ax.axhline(2.5, color='green', linewidth=2, label='Decision boundary')
ax.fill_between(x_line, 0, 2.5, alpha=0.1, color='#3498db')
ax.fill_between(x_line, 2.5, 17, alpha=0.1, color='#e74c3c')
ax.set_xlabel('x'); ax.set_ylabel('x²')
ax.set_title('2D Feature Space φ(x)=[x, x²]\\nLINEARLY separable!', fontsize=10)
ax.legend(fontsize=8)

# Panel 3: Back in 1D — nonlinear boundary
ax = axes[2]
ax.scatter(x_neg, np.zeros_like(x_neg), c='#e74c3c', s=50)
ax.scatter(x_pos, np.zeros_like(x_pos), c='#3498db', s=50)
ax.axvline(-np.sqrt(2.5), color='green', linewidth=2, linestyle='--', label='SVM boundary')
ax.axvline( np.sqrt(2.5), color='green', linewidth=2, linestyle='--')
ax.fill_betweenx([-0.5, 0.5], -np.sqrt(2.5), np.sqrt(2.5), alpha=0.15, color='#3498db', label='Predicted +1')
ax.set_ylim(-0.5, 0.5); ax.set_xlabel('x')
ax.set_title('1D: SVM with Polynomial Kernel\\nLearned nonlinear boundary', fontsize=10)
ax.legend(fontsize=8)

plt.suptitle('The Kernel Trick: Map to Higher Dimensions → Linear Separator', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('svm_kernel_trick.png', dpi=100, bbox_inches='tight')
plt.show()

print("KEY INSIGHT:")
print("SVM only needs K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)")
print("We NEVER compute φ(x) explicitly — just evaluate the kernel function directly.")
print("RBF kernel: K(xᵢ,xⱼ) = exp(-γ‖xᵢ-xⱼ‖²)  → maps to ∞-dimensional space!")
print()
print("Interview formula: Margin = 2 / ||w||  (maximized by minimizing ||w||²/2)")
""")

# ── New Section: Cross-Validation ────────────────────────────────────────────
CV_MD = md_cell("""## 2.13 Cross-Validation

**Listed in Amazon companion document. K-Fold + Stratified + Time Series CV.**

### WHY cross-validation:
A single train/val split depends on the random seed — you might get lucky (easy val) or unlucky (hard val). CV averages over K different splits for a **reliable estimate** of generalization error.

### K-Fold CV:
```
Dataset split into K equal folds.
For k = 1 to K:
    Train on folds {1,...,K} \\ {k}
    Validate on fold k
Average the K validation scores.

Result: K-1/K of data used for training in each fold.
```

### Stratified K-Fold:
Like K-Fold, but each fold preserves the **class proportion** from the full dataset.
Use when: imbalanced classes (e.g., 95% negative, 5% positive — regular split might get 0% positives in a fold).

### [REAL Q] "Why NOT use cross-validation on YOUR BLaIR project?"
> "Two reasons. First: compute cost. Each BERT fine-tuning epoch takes ~3.5 hours on Kaggle GPU. 5-fold CV = 5× full training = ~75 hours. Not feasible for a solo project. Second: we have temporal and product-level structure — product B0052J9UT6 belongs in exactly one split. We use a single product-level train/val/test split instead, which is the standard for the retrieval literature. McNemar significance tests across 9,972 test queries compensate for the lack of CV."

### Time Series CV (Walk-Forward Validation):
```
Fold 1: Train on t=1..100,  Validate on t=101..120
Fold 2: Train on t=1..120,  Validate on t=121..140
Fold 3: Train on t=1..140,  Validate on t=141..160
```
NEVER use future data to predict past. Used in H&M project.

### Hyperparameter Tuning with CV:
```
for C in [0.01, 0.1, 1, 10]:
    cv_scores = []
    for fold in k_folds:
        train, val = fold
        model = SVM(C=C)
        model.fit(train)
        cv_scores.append(model.score(val))
    mean_score[C] = mean(cv_scores)
Best C = argmax mean_score
```

### Nested CV (Two loops — unbiased model selection):
```
Outer loop: K folds for model evaluation
  Inner loop: K' folds for hyperparameter selection
  
Without nested CV: optimistic bias (you selected hyperparams on same data you evaluate)
```
""")

CV_CODE = code_cell("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Visualize K-Fold vs Stratified K-Fold ────────────────────────────────────

K = 5
N = 30  # samples

fig, axes = plt.subplots(3, 1, figsize=(12, 7))

# ── Standard K-Fold ──────────────────────────────────────────────────────────
ax = axes[0]
fold_size = N // K
for fold in range(K):
    val_start = fold * fold_size
    val_end   = val_start + fold_size
    
    # Train portions
    ax.barh(fold, val_start, left=0, color='#3498db', height=0.5, label='Train' if fold==0 else '')
    ax.barh(fold, N - val_end, left=val_end, color='#3498db', height=0.5)
    # Validation portion
    ax.barh(fold, fold_size, left=val_start, color='#e74c3c', height=0.5, 
            label='Validation' if fold==0 else '')
    ax.text(val_start + fold_size/2, fold, f'Val', ha='center', va='center', 
            color='white', fontsize=8, fontweight='bold')

ax.set_yticks(range(K)); ax.set_yticklabels([f'Fold {k+1}' for k in range(K)])
ax.set_xlabel('Sample index'); ax.set_title(f'{K}-Fold Cross-Validation', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)

# ── Stratified K-Fold ────────────────────────────────────────────────────────
ax = axes[1]
# Show class balance maintained per fold
classes = [0, 1]
class_counts = [0.8, 0.2]  # 80% class 0, 20% class 1

for fold in range(K):
    # Class 0 portion per fold
    n0 = class_counts[0] * N / K
    n1 = class_counts[1] * N / K
    x_start = fold * N / K
    ax.barh(0, n0, left=x_start, color='#3498db', height=0.5, 
            label='Class 0 (80%)' if fold==0 else '', edgecolor='white')
    ax.barh(0, n1, left=x_start + n0, color='#e74c3c', height=0.5,
            label='Class 1 (20%)' if fold==0 else '', edgecolor='white')
    ax.axvline(fold * N/K, color='gray', linewidth=1, alpha=0.5)
    ax.text(x_start + N/K/2, 0, f'F{fold+1}\\n{int(n0)}:{int(n1)}', 
            ha='center', va='center', fontsize=7, color='black')

ax.set_yticks([0]); ax.set_yticklabels(['Each fold'])
ax.set_title('Stratified K-Fold — Each fold preserves 80:20 class ratio', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(-0.4, 0.4)

# ── Time Series Walk-Forward ─────────────────────────────────────────────────
ax = axes[2]
n_folds = 4
window = 15
step = 5

for fold in range(n_folds):
    train_end = 20 + fold * step
    val_start = train_end
    val_end   = val_start + step
    
    ax.barh(fold, train_end, left=0, color='#3498db', height=0.5)
    ax.barh(fold, step, left=val_start, color='#e74c3c', height=0.5)
    ax.text(train_end/2, fold, f'Train (t=1..{train_end})', 
            ha='center', va='center', color='white', fontsize=7)
    ax.text(val_start + step/2, fold, f'Val\\nt={val_start+1}..{val_end}', 
            ha='center', va='center', color='white', fontsize=6)

ax.set_yticks(range(n_folds)); ax.set_yticklabels([f'Fold {k+1}' for k in range(n_folds)])
ax.set_title('Time Series Walk-Forward Validation (H&M Project)\\nNEVER use future data to predict past!', fontweight='bold')

plt.tight_layout()
plt.savefig('cross_validation.png', dpi=100, bbox_inches='tight')
plt.show()

# Score variance demo
np.random.seed(42)
scores = np.random.normal(0.82, 0.04, 5)
print(f"K-Fold CV scores: {[f'{s:.4f}' for s in scores]}")
print(f"Mean ± Std: {scores.mean():.4f} ± {scores.std():.4f}")
print()
print("Why NOT CV on BLaIR:")
print("  Each BERT epoch ≈ 3.5 hours × 15 epochs × 5 folds = 262 hours!")
print("  Instead: single product-level split + McNemar on 9,972 test queries")
""")

# ── Load and patch notebook ───────────────────────────────────────────────────
with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# 1. Update Table of Contents (Cell 0) ─────────────────────────────────────
# Add KNN, SVM, Cross-validation and Why Amazon to TOC
old_toc_line = "| 2.11 | KNN — K Nearest Neighbors |\n"
new_toc_block = """| 2.11 | KNN — K Nearest Neighbors (Full Theory + Decision Boundaries) |
| 2.12 | SVM + Kernel Trick (linear → RBF → polynomial) |
| 2.13 | Cross-Validation (K-Fold, Stratified, Time Series) |
"""

# Update ToC rows for 2.11-2.22 and add Why Amazon
for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "2.11 | RNN" in src or "2.11 | KNN" in src:
            # Rebuild the section 2 TOC rows
            new_src = src.replace(
                "| 2.11 | RNN & LSTM |",
                "| 2.11 | KNN — K Nearest Neighbors (Theory + Boundary Visualization) |\n| 2.12 | SVM + Kernel Trick (margin maximization, RBF, polynomial) |\n| 2.13 | Cross-Validation (K-Fold, Stratified, Time-Series) |\n| 2.14 | RNN & LSTM |"
            ).replace(
                "| 2.12 | SVM — Support Vector Machines |",
                "| 2.15 | SVM — Support Vector Machines (already merged into 2.12) |"
            ).replace(
                "| **PART 5** | LEADERSHIP PRINCIPLES (STAR Stories) |",
                "| **PART 5** | LEADERSHIP PRINCIPLES (STAR Stories) |\n| **WHY** | Why Amazon — 60-second memorized answer |"
            )
            cells[i]["source"] = new_src
            print(f"  Updated TOC at cell {i}")
            break

# 2. Find insertion point: after 2.9 Bayes cell, before 2.10 F-Beta ─────────
# We insert KNN, SVM, CV between "2.10 F-Beta" and "2.11 RNN"
insert_before_cell = None
for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "## 2.11 RNN" in src:
            insert_before_cell = i
            break

if insert_before_cell is None:
    # Fallback: insert before Part 3
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            if "## 3.1 BLaIR" in src:
                insert_before_cell = i
                break

print(f"  Inserting KNN, SVM, CV before cell {insert_before_cell}")

# Build insertion list: KNN (md+code), SVM (md+code), CV (md+code)
new_cells = [KNN_MD, KNN_CODE, SVM_MD, SVM_CODE, CV_MD, CV_CODE]

# Insert at position
for j, nc in enumerate(new_cells):
    cells.insert(insert_before_cell + j, nc)

nb["cells"] = cells

# 3. Save patched notebook ────────────────────────────────────────────────────
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"\n✅ Patched notebook saved.")
print(f"   Total cells: {len(nb['cells'])}")
print(f"   Added: KNN theory, KNN code, SVM+kernel, SVM code, Cross-validation, CV code")
