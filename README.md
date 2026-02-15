# Hybrid-GNNxTranformer: Structural Graph-Transformer for AI Code Detection

![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![Status](https://img.shields.io/badge/Status-Research_Prototype-blue?style=for-the-badge)

"This model utilizes the Graph Attention Network architecture proposed by Velickovic et al. (2018) and the standard Transformer Encoder architecture (Vaswani et al., 2017)."

A hybrid deep learning architecture that detects AI-generated code by analyzing both **sequential logic** (Transformer) and **structural syntax patterns** (Graph Attention Network).
Currently on a very small dataset.
Process:
1) Instead of initiating both Gat and Transformer train together, we freeze Transformer gradients in the initial phase, so that it does not dominate the whole training.
2) This allows GAT to capture global structural/graphical patterns in the data.
3) After a few epochs, we unfreeze the Transformer and merge the GAT and Transformer training.
4) Then we make another module for Transformer alone, and we run validation tests on both GAT+Transformer and Transformer alone for the same epochs.
5) We observe that GAT+Transformer shows better accuracy on validation tests on the same number of epochs.

Unlike standard text classifiers, Hybrid-GNNxTransformer constructs a **Global Co-occurrence Graph** of the code vocabulary, allowing it to "see" the shape of programming syntax, not just the order of words.

---

## 🏆 Key Results (Ablation Study)

Our experiments demonstrate that treating code purely as text (Transformer only) is insufficient for distinguishing subtle AI patterns. Injecting structural graph knowledge yields a **~19.5% performance boost**.

Analysis from a Kaggle dataset: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text(IN BIG DATA BRANCH).
| Model Architecture | Accuracy | Insight |
| :--- | :--- | :--- |
| **Transformer Only** (Ablated) | 37.03% | *Failed to learn meaningful patterns* |
| **Hybrid (Transformer + GAT)** | **91.79%** | *Recovered structure & syntax logic* |
| **Structural Contribution** | **+54.77%** | **The graph is the primary decision maker** |

> **Note:** Final evaluation on the full test set (975 samples) achieved **93.54% Accuracy**.
### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Human Code** | 96.5% | 93.2% | 0.95 | 614 |
| **AI Code** | 89.0% | 94.2% | 0.92 | 361 |

### Confusion Matrix
| | **Predicted Human** | **Predicted AI** |
| :--- | :---: | :---: |
| **Actual Human** | **572** (True Neg) | 42 (False Pos) |
| **Actual AI** | 21 (False Neg) | **340** (True Pos) |

* **Low False Negative Rate:** Only 21 out of 361 AI samples slipped through undetected.
* **High Fidelity:** The model correctly identified 96.5% of human code, minimizing frustration for developers.

---

---

## 🧠 Model Architecture

The model utilizes a dual-branch fusion strategy:



[Image of neural network architecture diagram]


1.  **Semantic Branch (Transformer Encoder):**
    * Processes code as a sequence of tokens.
    * Captures variable naming conventions, logic flow, and sequential dependencies.
    * *Result:* A dense vector representing the "meaning" of the code.

2.  **Structural Branch (VocabGAT):**
    * Constructs a sparse **Token Co-occurrence Graph** from the training corpus.
    * A **Graph Attention Network (GAT)** learns embeddings based on neighbor relationships (e.g., how often `static` connects to `void` vs `class`).
    * *Result:* A dense vector representing the "syntax topology" of the code.

3.  **Adaptive Gated Fusion:**
    * Instead of simple concatenation, a learnable gate ($g$) dynamically weights the importance of structure vs. meaning:
    $$Embedding = (1 - g) \cdot X_{semantic} + g \cdot X_{structural}$$

---

## 📂 Dataset Format
Two datasets: 1) Codes,      2) Textual

## For code dataset:
The model requires a CSV file with the following columns:
* `text`: The raw source code string.
* `ai`: Binary label (`0` = Human-written, `1` = AI-generated).

*Note: The project automatically handles tokenization of special structural characters (e.g., `{`, `}`, `;`) to preserve code syntax.*

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Shauryam22/Hybrid-GNN-x-Transformer.git
cd Hybrid-GNN-x-Transformer
pip install torch pandas scikit-learn networkx matplotlib
