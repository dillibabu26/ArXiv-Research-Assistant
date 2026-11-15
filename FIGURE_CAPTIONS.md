# Figure captions for REPORT.md

Short, factual captions using the values measured on your laptop run (11 Nov 2026).
They describe what each figure shows; the *interpretation* belongs in the report text, in
your own words. Check your brief for whether captions count towards the word limit.

> REPORT.md (rubric-structured version) uses its own numbering: Figure 1 architecture,
> Figure 2 class imbalance, Figure 3 SVM confusion matrix; Appendix A holds top categories,
> word frequency, title ROUGE and abstract length. Captions below remain valid as descriptions.

Insert a figure in Markdown with: `![Figure 1](outputs/figures/01_top_categories.png)`

---

## Data exploration

**Figure 1** (`01_top_categories.png`). The 15 most frequent primary categories among 154,911
cleaned arXiv papers first submitted in 2019. cs.LG and cs.CV are more than twice the size of any
other category; the ten largest (used as classes) account for 41.5% of papers.

**Figure 2** (`02_abstract_length.png`). Abstract length in words (median 146; 5th–95th percentile
54–257). The dotted line marks the 20-word minimum used in cleaning.

**Figure 3** (`03_title_length.png`). Title length in words (median 9; 5th–95th percentile 5–16),
about one-fifteenth of the abstract length.

**Figure 4** (`04_missing_values.png`). Share of records with each field missing or empty. All
modelling fields are complete; `doi`, `update_date` and `comments` are sparse and were not used.

**Figure 5** (`05_most_frequent_words.png`). The 20 most frequent unigrams and bigrams in the
41,892-abstract modelling sample after stop-word removal.

**Figure 6** (`06_category_imbalance.png`). Left: class sizes before and after capping at 5,000
papers per class (imbalance ratio 2.6:1 → 1.5:1). Right: number of categories listed per paper;
53.5% of papers are cross-listed.

**Figure 7** (`07_word_frequency_analysis.png`). Left: term frequency against rank on log-log axes
(Zipf plot). Right: share of all tokens covered by the most frequent *n* terms; 20,000 and 50,000
terms cover 95.7% and 98.5% respectively.

## Results

**Figure 8** (`classification_comparison.png`). Accuracy, macro-F1 and weighted F1 of the three
TF-IDF classifiers on the held-out test set (n = 6,284).

**Figure 9** (`confusion_matrix_linear_svm.png`). Row-normalised confusion matrix of the Linear SVM
on the test set; each row shows how papers of one true category were classified. The largest
confusions are cond-mat.mes-hall ↔ cond-mat.mtrl-sci and cs.LG ↔ cs.CV.
*(The Naive Bayes and Logistic Regression matrices show the same pattern; put them in an appendix
or mention them in the text rather than including all three.)*

**Figure 10** (`comparison_classification.png`). Accuracy and macro-F1 of all classifiers on the
same 300 class-balanced test papers used for the model comparison.

**Figure 11** (`comparison_inference_time.png`). Median time to classify a single abstract,
including text cleaning, on a laptop CPU. Batch throughput is similar for all three models
(0.21–0.23 ms per abstract).

**Figure 12** (`comparison_titles.png`). ROUGE-1, ROUGE-2 and ROUGE-L F1 between generated and
original titles for 300 test papers. FLAN-T5 was used zero-shot; the baselines copy a sentence
from the abstract, truncated to 12 words.

---

### Suggested selection if space is tight
A 1,500-word report usually has room for about 5–7 figures. A strong minimal set:
Figure 1 (data), Figure 6 (imbalance and cross-listing), Figure 7 (vocabulary → TF-IDF settings),
Figure 9 (errors), Figure 12 (titles). Report the classification numbers in Tables 1–2 rather than
as Figures 8 and 10, which repeat the same values.
