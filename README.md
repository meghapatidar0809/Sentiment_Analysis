# Sentiment Analysis using Multinomial Naive Bayes and Active Learning

This repository contains the implementation of a **Multinomial Naive Bayes classifier** for sentiment analysis on the **Sentiment140 dataset**. Additionally, it includes an **Active Learning Strategy (ALS)** to selectively label the most informative data points to improve model performance with fewer labeled samples.

---

## Dataset

The dataset used is the **Sentiment140 dataset**, which consists of approximately **1.593 million tweets**. Each tweet is labeled with a sentiment score ranging from **1 to 5**, which has been converted to binary labels (**0 for negative** and **1 for positive sentiment**) for simplicity.

---

## Preprocessing

The text data has been preprocessed to:
- Remove **stop-words**, **URLs**, and **Twitter handles**.
- Apply **stemming** and **lemmatization** to each word in the text.
- Remove all **blank texts** from the corpus.

---

## Implementation

### 1. Count Vectorizer

The `utils.py` file contains the implementation of a **Count Vectorizer**, which converts each sentence into a **k-dimensional vector**. Each dimension corresponds to a word in the vocabulary, and the value in each dimension represents the **count of the corresponding word** in the sentence.

**Caution:** Since the vocabulary is large, **sparse matrices** are used to store the count vectors efficiently. This avoids memory overflow errors.

### 2. Multinomial Naive Bayes

The `model.py` file contains the implementation of the **Multinomial Naive Bayes classifier**. The classifier uses the **class conditional probabilities** of each word to determine the sentiment of a given sentence.

### 3. Active Learning Strategy

The `prob2.py` file contains the implementation of the **Active Learning Strategy (ALS)**. The strategy selects the **most uncertain data points** for labeling to improve the model's performance with fewer labeled samples.

---

## Usage

### Training and Testing the Model

To train and test the Multinomial Naive Bayes classifier, run the following command:

```bash
python prob1.py --sr no <5 digit SR#>
```

---

## Active Learning Strategy

To run the Active Learning Strategy, execute the following command:

```bash
python prob2.py --sr_no <5-digit SR#> --run_id <int> [--is_active]
```

---

## Random Strategy

To run the Active Learning Strategy, execute the following command:

```bash
python prob2.py --sr_no <5-digit SR#> --run_id <int> 
```

---

## Generating Plots

After running the Active Learning Strategy and Random Strategy generate the required plots by executing:

```bash
python plot.py --sr_no <5-digit SR#> --supervised_accuracy <float>
```

---

## Requirements

To run this project, ensure you have the following installed:

- Python 3.x
- NumPy
- SciPy
- scikit-learn
- Matplotlib

---

## References

- [Sentiment140 Dataset](https://huggingface.co/datasets/stanfordnlp/sentiment140)
- [Count Vectorizer](https://github.com/yashika51/Understanding-Count-Vectorizer)
- [Stemming and Lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
- [TF-IDF Vectorizer](https://en.wikipedia.org/wiki/Tf-idf)
- [Gini Impurity and Entropy](https://www.geeksforgeeks.org/gini-impurity-and-entropy-in-decision-tree-ml/)
