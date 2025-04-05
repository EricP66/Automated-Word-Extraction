# Automated Word Extraction for Bias Detection in Educational Texts

This project implements an automated method to extract and evaluate demographic **target** and **attribute** word sets from AI-generated educational texts.

We leverage prompt engineering and contextualized bias assessment using CEAT (Contextualized Embedding Association Test) to detect **demographic biases** such as gender, race, and nationality in generative educational content.

---

## 📌 Overview of the Methodology (as in the paper)

The pipeline consists of:

1. **Manual rubric-based ground-truth extraction** of target/attribute words  
2. **Automated extraction** using GPT-4o + Retrieval-Augmented Generation
3. **Bias evaluation** using the CEAT framework:
   - Effect size computation
   - Combined effect size (CES)
   - Cosine similarity between auto and manual sets
4. **Validation** using Pearson correlation between scores

---

## 📂 Repository Structure

### `Codes/`  
Contains Python scripts and Jupyter notebooks for word extraction and evaluation:

- `Prompt_Word_Extractor.ipynb`:  
  Based on the carefully crafted prompts, automatically extract the most appropriate **target** and **attribute** word sets we desire.

- `Embeddings_docx.py`:  
  Important pre-requisite to compute CEAT scores for word sets. This step successfully converts `.docx` texts into contextualized embeddings using OpenAI models.
  
- `Score.py`:  
  Generates CEAT effect size scores based on the CEAT equations in Section 2.2.
  
- `Similarity.py`:  
  Computes cosine similarity between automatically and manually extracted word sets. Matches **semantic alignment** metrics from Section 3.1 of the paper.

- `Pearson.py`:  
  Calculates Pearson correlation (used to validate consistency between automated vs. manual CEAT scores as in Section 3.2).

---

### `Engineered Prompts/`  
Prompt templates used in GPT-4o for few-shot word extraction. Each includes:

- **Bias type examples** (e.g., gender, race, nationality — see **Table 1** in the paper)
- **Explicit constraints** (e.g., avoid inferred terms, include all demographic groups, locate words only available in educational texts etc.)
  
These are used in the **automated word extraction**.

---

### `Rubrics/`  
Manual ground-truth rubrics used for human-annotated word set construction.

- Based on **Table 2** in the paper:  
  Important steps include:
  1). Explicit/implicit demographic identification;
  2). Contextual filtering;
  3). Avoiding redundancy;
  4). Avoid over-Specificity;
  5). Avoid broad items;
  etc.

---

### `Results/`  
CEAT evaluation outputs for each course:

- Files named like `ceat_results_course1_auto.csv` and `ceat_results_course1_gt.csv`  
  Correspond to **automated vs ground-truth results**.
- CEAT effect size scores from both sets can be compared directly (see **Table 4** in paper).

---

### `Texts-for-Rubrics/`  
AI-generated educational biased texts used to determine our ground-truth rubrics and derive the **ground-truth word sets**.

---

### `Texts-for-Testing/`  
More educational texts used to **validate** the automated extraction pipeline.

---

## 📊 Evaluation Metrics

- **Cosine Similarity** between word sets  
  Used to quantify alignment between auto and ground-truth (see **Table 3**)
  
- **Pearson's r** between CEAT scores  
  Used to evaluate the reliability of the automated system (see **Section 3.2**, r = 0.9930 indicates high correlation)

---

## 📌 Keywords
*Bias Detection · CEAT · Prompt Engineering · RAG · Educational NLP · Fairness in GenAI · Word Extraction*

