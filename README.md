## 🔮 LLM Configuration – Understanding `tokens` and `temperature` and `RAG AND Langchain`

This application uses Large Language Models (LLMs) to generate contextual answers based on retrieved documents. Two critical parameters that directly influence the behavior of LLMs are:

---

### 🧩 1. Tokens

* LLMs process text in the form of **tokens**, not raw characters or words.
* A token can be:

  * A **word**
  * A **part of a word**
  * **Punctuation**
  * Even **whitespace**
* Example:

  * The sentence `"What’s going on?"` might be split into tokens like `['What', '’s', 'going', 'on', '?']`
* LLMs predict **one token at a time**, based on the context of all previous tokens.

### 🧠 Max Tokens

* `max_tokens` defines the **maximum number of tokens** the LLM is allowed to generate in response.
* Affects:

  * **Length** of the response
  * **Computation cost**
* Example:

  * If `max_tokens=100`, the model will stop after generating 100 tokens or earlier.

Use smaller values for faster, shorter answers. Use larger values when expecting longer or detailed output.

---

### 🎲 2. Temperature

* `temperature` controls the **randomness** of the LLM's responses.
* It adjusts the **probability distribution** from which the next token is sampled.

#### 🔹 Low Temperature (e.g., 0.2)

* **More deterministic and repetitive** outputs
* Prioritizes the **most likely next token**
* Best for tasks like factual Q\&A, summarization, and deterministic logic

#### 🔹 High Temperature (e.g., 1.2–1.5)

* **More random and diverse** outputs
* Encourages creativity, storytelling, ideation
* May occasionally produce irrelevant or surprising responses

#### 🔹 Temperature = 1.0

* Samples tokens based on their **true probabilities** (unadjusted)
* Balanced behavior: some creativity, some coherence

---

### 📚 Summary of Learnings

| Parameter     | Description                                  | Recommended Usage                                 |
| ------------- | -------------------------------------------- | ------------------------------------------------- |
| `max_tokens`  | Maximum number of tokens in generated output | 100–300 for short answers, 500+ for detailed ones |
| `temperature` | Controls randomness/creativity in output     | 0.2 for precise Q\&A, 1.0+ for creativity         |

---

Feel free to adjust these values in the UI sliders to fine-tune how the model behaves!
