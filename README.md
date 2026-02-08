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

## Usage

### Install Dependencies
```bash
pip install torch tiktoken datasets transformers
```

### Train
```bash
python -m nanoJEPA.train
```

### Inspection
Check `inspect_checkpoint.py` to view model weights and ensuring no NaNs.

## Success Metrics
- **Token Loss**: Ensures the model remains a competent language model.
- **JEPA Loss**: Measures how well `[PRED]` approximates the semantic content of the Answer. 
- **Latent Similarity**: `[PRED]` should be closer to `Answer` latent than random vectors.