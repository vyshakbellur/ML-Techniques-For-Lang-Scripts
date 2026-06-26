# FeedelLigence: Cross-Modal Script Phylogenetics

*A deep learning framework for quantifying morphological and acoustic similarity across historical writing systems. Published in **Digital Scholarship in the Humanities** (Oxford University Press).*

---

## Abstract

Writing systems are among the most durable artifacts of human civilization. **FeedelLigence** provides an objective, quantitative, and reproducible framework for measuring morphological relationships between historical scripts using deep neural networks, bypassing the subjectivity inherent in traditional visual paleographic analysis.

The core model — a **Siamese Convolutional Neural Network** — learns a distance metric between character images by training on over 28,000 Ethiopic (Ge'ez) script specimens. The resulting embedding space encodes structural and geometric primitives (strokes, curves, junctions) common across scripts. By embedding characters from target scripts into this shared space, we can compute **phylogenetic distances** between writing systems without requiring parallel corpora or labeled cross-script pairs.

**Published finding**: Armenian shows a statistically closer morphological relationship to ancient Ethiopic than previously established by traditional linguistic analysis — suggesting a shared visual ancestor or transmission pathway.

---

## Architecture

```mermaid
graph TD
    subgraph Visual Encoder (Siamese CNN)
        A[Glyph Image 1 64x64] --> C(CNN Backbone)
        B[Glyph Image 2 64x64] --> C
        C --> D[128-dim Embedding Space]
    end

    subgraph Multimodal Extension
        D --> E{Contrastive Loss}
        F[Phoneme String GRU Encoder] --> G[128-dim Phonetic Embedding]
        G --> H{CLIP-style Cross-Modal Alignment}
        D --> H
    end

    subgraph Phylogenetic Evaluation
        D --> I[Pairwise Distance Matrix]
        G --> I
        I --> J[Script Similarity Tree via UMAP / Hierarchical Clustering]
    end
```

---

## Research Questions

| Priority | Question |
|---|---|
| Q1 | Does the model confirm Ethiopic–Armenian morphological affinity at scale? |
| Q2 | Does affinity hold for Sanskrit, Phoenician, Sumerian cuneiform? |
| Q3 | Are there **bridge scripts** (Brahmi, Coptic) that act as phylogenetic intermediaries? |
| Q4 | Do morphologically similar scripts share acoustic phoneme proximity? |
| Q5 | Can a joint visual-acoustic embedding outperform lexical phylogenetics? |

---

## Installation

```bash
git clone git@github.com:vyshakbellur/ML-Techniques-For-Lang-Scripts.git
cd ML-Techniques-For-Lang-Scripts
pip install .
```

## Execution

### Train the Siamese Visual Encoder
```bash
python train.py
```

### Evaluate Cross-Script Similarity
```bash
python evaluate_similarity.py
```

### Evaluate Multimodal Phylogenetic Distance (Visual + Phonetic)
```bash
python evaluate_phylogeny.py
```

---

*Submitted to: DSH (Oxford), ACL, COLING, CHR — Computational Humanities Research.*
