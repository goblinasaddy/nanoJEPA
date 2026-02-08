# nanoJEPA (Educational LLM-JEPA)

A minimal, educational implementation of **Joint Embedding Predictive Architectures (JEPA)** for language, applied to the GSM8K dataset.

Inspired by:
- **nanoGPT** for simplicity and clarity.
- **LLM-JEPA** for the conceptual framework.
- **GSM8K** as a reasoning-heavy dataset.

## Core Concept
Standard LLMs predict the *next token*. 
nanoJEPA predicts the **latent representation of the answer** from the **latent representation of the question**.

### Architecture
![nanoJEPA Architecture](nanoJEPA arch.png)
*(Please save the provided architecture image as `nanoJEPA_architecture.png` in this directory)*

- **Backbone**: Single decoder-only Transformer (GPT-2 style).
- **Views**: 
  - `View A`: Question (e.g., math problem).
  - `View B`: Answer (numerical result).
- **Latent Prediction token `[PRED]`**:
  - Attends to the Question.
  - Predicts the final hidden state of the Answer.
  - Does **not** see the Answer tokens (masked out).

### Attention Masking (The "Secret Sauce")
To enforce JEPA constraints in a single Transformer:
1. **Q -> Q**: Standard causal masking.
2. **A -> A**: Standard causal masking (independent of Q).
3. **[PRED] -> Q**: Full attention to Question.
4. **[PRED] -/-> A**: Blocked. The model must *predict* A's content.

## File Structure
- `config.py`: Hyperparameters.
- `data.py`: GSM8K loading and tokenization.
- `model.py`: Transformer with custom JEPA masking.
- `train.py`: Training loop with Token Loss + JEPA Loss.
- `demo.py`: **Interactive Gradio Interface**.
- `eval_alignment.py`: Script to verify latent alignment.

## Usage

### 1. Install Dependencies
```bash
pip install torch tiktoken datasets transformers gradio matplotlib
```

### 2. Train
```bash
python -m nanoJEPA.train
```
*Note: The default run is short (educational). For better results, increase `max_iters` in `config.py`.*

### 3. Interactive Demo (Gradio)
Launch a web interface to test the model:
```bash
python demo.py
```
*The model uses greedy decoding to project the latent prediction into text.*

### 4. Verify Latent Alignment
Run the evaluation script to compare JEPA vs Baseline:
```bash
python eval_alignment.py
```
This generates `latent_alignment.png`, showing how the cosine similarity between `[PRED]` and `Answer` improves over time.

## Success Metrics
- **Token Loss**: Ensures the model remains a competent language model.
- **JEPA Loss**: Measures how well `[PRED]` approximates the semantic content of the Answer. 
- **Latent Similarity**: `[PRED]` should be closer to `Answer` latent than random vectors.