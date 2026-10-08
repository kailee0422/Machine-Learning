# Final Project: Real-Time Email Monitoring and Alert System Based on a Self-Trained Spam Detection Model

## Introduction

While Gmail and other providers offer built-in spam detection, it isn't foolproof and
often misses sophisticated phishing attempts. This project builds an email monitoring
system around a **custom-trained spam detection model**: it fetches emails over IMAP,
classifies each one, and — if flagged as spam or phishing — sends the user a warning
email over SMTP.

## Structure

```
Connect to the mail server → Check and extract new emails → Detect spam by model → Send warning email → Repeat
```

- **Connect**: `imaplib.IMAP4_SSL("imap.gmail.com")`, log in, confirm connection.
- **Check and extract**: select the mailbox folder, search for new messages, fetch each
  one's subject/body.
- **Detect spam**: Chinese subject/body text is translated to English via
  `googletrans`, vectorized, and passed to the trained classifier.
- **Send warning email**: if flagged, send an SMTP warning email (with a "beware of
  scams" graphic) back to the user.

## Materials and Methods

**Dataset**: [Spam Email](https://www.kaggle.com/) (5,157 messages, 87% ham / 13% spam).

**Text representations compared** (each fed into every model below):

| Method | Advantages | Disadvantages |
|---|---|---|
| Word2idx | Simple, fast, memory-efficient | No semantic understanding, fixed vocabulary |
| Hashing Vectorizer | Efficient for large datasets, fixed-size vectors | Potential hash collisions, no semantic relationships |
| Word2Vec (skip-gram) | Captures semantic relationships, pretrained models available | Needs large training data, context-independent |
| BERT-base-uncased | Context-aware, handles subwords, versatile | Computationally intensive, high memory usage |

**Models compared**: LSTM (bidirectional, embedding dim 64 → hidden 128, see
[`lstm_model_graph.dot`](lstm_model_graph.dot) for the full computation graph), Decision
Tree, and Logistic Regression — implemented in [`fetch_mail.ipynb`](fetch_mail.ipynb),
[`decision_tree.ipynb`](decision_tree.ipynb), and [`LogisticRegression.ipynb`](LogisticRegression.ipynb)
respectively, all trained on [`spam.csv`](spam.csv) (80/20 train/val split).

## Results

Validation precision / recall / F1 across all four text representations:

| Representation | LSTM | Decision Tree | Logistic Regression |
|---|---|---|---|
| Word2idx | 95.98% | 86.90% | 83.24% |
| Hashing Vectorizer | 98.48% | 96.58% | 96.99% |
| Word2Vec | 88.97% | 93.95% | 95.41% |
| **BERT-base-uncased** | **99.46%** | 96.21% | 99.11% |

**LSTM + BERT-base-uncased** was the best overall combination (99.46% val F1), and was
chosen as the final production model.

## Conclusion

Combining an **LSTM** classifier with **BERT-base-uncased** as the tokenizer gave the
strongest spam/phishing detection across every text-representation method tested. With
IMAP for fetching and SMTP for alerting, the resulting system can monitor a mailbox and
notify the user in real time. Future work: incorporate a larger, self-collected dataset
for further training.

## Code

- [`fetch_mail.ipynb`](fetch_mail.ipynb) — email fetching/translation pipeline and the LSTM model
- [`LogisticRegression.ipynb`](LogisticRegression.ipynb) — logistic regression baseline
- [`decision_tree.ipynb`](decision_tree.ipynb) — decision tree baseline

Full project proposal: [`Midterm_Proposal.pdf`](Midterm_Proposal.pdf). Full results
writeup: [`Final_Report.pdf`](Final_Report.pdf).
