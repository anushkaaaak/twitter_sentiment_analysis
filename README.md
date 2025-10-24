#  Twitter Sentiment Analysis: DistilBERT Fine-Tuning

### **Project Status: Complete | Accuracy: 96.50%**

---

##  Project Overview & Business Impact

This project demonstrates an **end-to-end  pipeline** for real-time social media sentiment classification. By leveraging a lightweight, state-of-the-art **Transformer model (DistilBERT)**, the solution achieves production-level accuracy on a large-scale, multi-class classification task (Positive, Negative, Neutral).

### **Key Achievements**

| Metric | Result | Impact | 
| :--- | :--- | :--- | 
| **Model** | **DistilBERT** (Transformer) | Utilized SOTA (State-of-the-Art) architecture for superior contextual understanding compared to traditional models. | 
| **Data Scale** | **~120,000 Tweets** | Successfully handled and processed a large dataset, demonstrating efficiency in data preparation. | 
| **Accuracy** | **96.50%** | Achieved high-fidelity classification, minimizing false positives/negatives in social listening. | 
| **Deployment Readiness** | **HuggingFace `Trainer` API** | Pipeline built for easy serialization (`.save_pretrained()`) 

---

##  Technology Stack

* **Deep Learning Framework:** **PyTorch**
* **NLP Library:** **HuggingFace Transformers** (DistilBERT, Tokenizer, Trainer)
* **Data Processing:** Pandas, NumPy, Python
* **Visualization:** Matplotlib, Seaborn, WordCloud

---

## Workflow Highlights (The `Twitter_Sentiment_Analysis_.ipynb` Notebook)

1.  **Massive Data Ingestion:** Processed large training (~103k) and validation (~16k) datasets.
2.  **Efficient Preprocessing:** Mapped labels and handled edge cases (NaNs, irrelevant entries) to ensure data quality.
3.  **Transformer-Native Tokenization:** Used `DistilBertTokenizerFast` to prepare inputs (`input_ids`, `attention_mask`) for the deep learning model.
4.  **Optimized Fine-Tuning:** Implemented the HuggingFace `Trainer` API with optimized `TrainingArguments` (e.g., `learning_rate=2e-5`, batching) for quick and effective transfer learning.
5.  **Comprehensive Evaluation:** Calculated standard multi-class metrics (**F1-Score, Precision, Recall, Accuracy**) and performed in-depth **WordCloud** analysis to visualize feature importance per sentiment class.

---

##  Key Visualizations: Data Exploration

This section presents the initial data exploration of the raw tweet text. While the Transformer model handles tokenization contextually, this analysis identifies the underlying topics and noise level in the dataset.

### **Word Cloud Analysis**

**Word Cloud for Negative Tweets**
![Word Cloud for Negative Tweets](assets/wordcloud_negative.png)

*The presence of terms like 'game', 'server', 'fix', and entity names clearly indicates negative sentiment is often driven by **product/service issues**.*

### **Top Uncleaned Tokens per Sentiment**

| **Negative (Count)** | **Positive (Count)** | **Neutral (Count)** |
| :--- | :--- | :--- |
| the (12493) | the (10163) | the (8731) |
| to (8685) | I (8011) | / (7493) |
| I (8632) | to (7501) | to (6776) |
| and (7967) | and (6458) | and (5892) |
| a (7049) | a (5480) | a (5267) |
| is (6317) | of (4962) | I (4774) |
| of (5472) | is (4190) | of (4710) |
| @ (4650) | for (4156) | for (3748) |
| in (4380) | in (3525) | . (3719) |
| for (4028) | this (2969) | in (3596) |
| my (3583) | my (2812) | is (3055) |
| this (3577) | it (2734) | on (2943) |
| you (3503) | on (2580) | @ (2376) |
| on (3486) | . (2528) | - (2317) |
| it (3449) | @ (2371) | my (2149) |

---

##  Future Scope (MLOps & Scaling)

* Implement the model using **PyTorch Lightning** for better code organization and experimentation.
* Containerize the trained model using **Docker** and deploy it as a REST API endpoint using **FastAPI** for low-latency inference.
* Integrate **MLflow** for robust experiment tracking and model versioning.
