下面这一整块就是完整 `README.md`，你可以直接复制粘贴到文件里用：

````markdown
# Fake News Detection with Evidence (DS340 Project)

This repository implements a fake–news detector for short news claims.
We compare several models (TF-IDF + Logistic Regression, Logistic Regression
with hand-crafted features, and BERT) and build a small web demo that shows
**prediction + supporting “evidence”** from a real-news corpus.

---

## 1. Project structure

At the repo root:

- `src/`
  - `config.py` – central paths for data and output
  - `prepare_main.py`, `prepare_covid.py`, `prepare_welfake.py` – clean each
    raw dataset
  - `merge_all_datasets.py` – combine all datasets into one `all` split
  - `add_features.py` – add numeric / linguistic features
  - `baseline_lr.py` – TF-IDF + Logistic Regression baselines
  - `baseline_lr_plus_feats.py` – TF-IDF + features + Logistic Regression
  - `bert_news_classifier.py` – DistilBERT classifier
  - `visualize_results.py` – generate all figures used in the report
  - `build_evidence_pairs_all.py` / `predict_lr_with_evidence_all.py`  
    – build REAL-news evidence and attach an evidence sentence to each
      prediction
  - `app_web.py` – local web demo (input a claim, get prediction + evidence)
  - `templates/index.html`, `static/style.css` – web UI
- `data/` – **raw data** (downloaded separately, see below)
- `output/`
  - `processed/` – cleaned CSVs (by dataset and by `all`)
  - `features/` – feature-engineered CSVs
  - `bert_all/` – saved BERT checkpoints
  - `evidence/` – REAL-news corpus and evidence pairs
  - `predictions/` – predictions with attached evidence
  - `results_summary.json` – compact summary of all results (used by
    `visualize_results.py`)

---

## 2. Data

Raw datasets live in **Google Drive** because of their size:

> https://drive.google.com/drive/u/1/folders/14tT2TWbL1fPqyDzV6FwPBfzMKnMkc3Xy

Download the **whole `data/` folder** from the link, and put it at the
project root so the structure looks like:

```text
DS340/
  data/
    main/
      Constraint_Train.csv
      Constraint_Test.csv
      Constraint_Val.csv
    covid/
      ...
    welfake/
      ...
  src/
    ...
````

The exact internal structure in Drive matches this repo.

---

## 3. Environment & dependencies

Tested with:

* Python 3.11
* Windows 11
* Optional: NVIDIA GPU with CUDA (for BERT and sentence-transformers)

Create a virtual environment and install dependencies:

```bash
# From the project root:
python -m venv .venv_torch
.\.venv_torch\Scripts\activate      # PowerShell on Windows

pip install --upgrade pip

# Core dependencies
pip install ^
  pandas numpy scikit-learn matplotlib seaborn tqdm ^
  flask ^
  sentence-transformers transformers

# Install PyTorch (CPU or GPU). For GPU with CUDA 12.x you can do:
# (or follow the official PyTorch install instructions)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you only want the classical Logistic Regression models and the web demo,
PyTorch / BERT are optional.

---

## 4. Data preprocessing pipeline

Run all commands from the project root **with the virtualenv activated**.

### 4.1 Clean each dataset

```bash
python src/prepare_main.py
python src/prepare_covid.py
python src/prepare_welfake.py
```

This writes cleaned CSVs into:

* `output/processed/main/`
* `output/processed/covid/`
* `output/processed/welfake/`

### 4.2 Build the combined `all` dataset

```bash
python src/merge_all_datasets.py
```

This creates `output/processed/all/train.csv`, `val.csv`, `test.csv`.

### 4.3 Add numeric / linguistic features

For each dataset (including `all`):

```bash
python src/add_features.py --dataset main
python src/add_features.py --dataset covid
python src/add_features.py --dataset welfake
python src/add_features.py --dataset all
```

Feature-augmented files go into `output/features/<dataset>/train_fe.csv`
and `val_fe.csv`.

---

## 5. Models

### 5.1 Logistic Regression baselines

#### (a) Simple TF-IDF + LR

```bash
python src/baseline_lr.py --dataset main
python src/baseline_lr.py --dataset covid
python src/baseline_lr.py --dataset welfake
python src/baseline_lr.py --dataset all
```

Each run prints:

* majority baseline
* TF-IDF + LR validation accuracy & macro F1
* per-class precision / recall

#### (b) TF-IDF + features + LR (main model)

```bash
python src/baseline_lr_plus_feats.py --dataset main
python src/baseline_lr_plus_feats.py --dataset covid
python src/baseline_lr_plus_feats.py --dataset welfake
python src/baseline_lr_plus_feats.py --dataset all
```

These correspond to the main numbers in the report (`lr_tfidf_feats`).

---

### 5.2 BERT (DistilBERT) classifier on `all`

BERT is trained only on the merged `all` dataset.

```bash
# Example (default hyperparameters)
python src/bert_news_classifier.py --dataset all
```

The script will:

* load `output/processed/all/{train,val}.csv`
* fine-tune `distilbert-base-uncased`
* save the best checkpoint into `output/bert_all/`
* print validation accuracy, macro F1 and a classification report
* sweep the decision threshold for predicting FAKE vs REAL

You can also **reuse** a trained checkpoint (no retraining) with:

```bash
python src/bert_news_classifier.py --dataset all --skip_train
```

(provided that `output/bert_all/` already contains a DistilBERT checkpoint).

---

## 6. Visualizations

Once you have run the LR & BERT scripts and updated `results_summary.json`,
you can generate all figures used in the report:

```bash
python src/visualize_results.py
```

This produces:

* Validation accuracy & macro-F1 by model × dataset
* Precision / recall per class (REAL vs FAKE)
* Threshold–vs–metric curves for BERT on `all`
* Label distribution plots

Figures are saved into `output/figures/` (or shown directly, depending on the
matplotlib backend).

---

## 7. Evidence retrieval & prediction with evidence

To attach **REAL-news evidence** to LR predictions on the combined `all`
dataset:

1. Build evidence pairs (nearest REAL news for each sample):

   ```bash
   python src/build_evidence_pairs_all.py
   ```

   This creates:

   * `output/evidence/all/all_pairs_train.csv`
   * `output/evidence/all/all_pairs_val.csv`
   * `output/evidence/all/all_pairs_test.csv`

2. Train the LR+features model on `all` and save predictions **with
   evidence**:

   ```bash
   python src/predict_lr_with_evidence_all.py
   ```

   Output:

   * Console: accuracy / macro-F1 for LR+features on `all` validation set
   * File: `output/predictions/all_lr_with_evidence_val.csv` containing

     ```text
     id, news_text, evidence_text, label, pred_label, pred_prob_fake, ...
     ```

   This CSV is what the web demo uses internally: every news snippet is
   paired with a retrieved REAL-news sentence that serves as “evidence”
   for the prediction.

---

## 8. Local web demo

The web app is a small Flask application that lets you paste a news claim
and see:

* predicted label (REAL / FAKE) from the LR+features model, and
* a retrieved REAL-news sentence as evidence.

1. Make sure you have:

   * run the preprocessing pipeline (Section 4),
   * run `predict_lr_with_evidence_all.py` at least once (Section 7).

2. Start the app:

   ```bash
   python src/app_web.py
   ```

3. Open a browser at:

   * [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. Type/paste a news snippet and click **Predict**.

   The page shows:

   * model prediction (REAL / FAKE)
   * predicted probability for FAKE
   * one nearest REAL-news sentence as evidence.

---

## 9. Reproducibility notes

* All paths are controlled by `src/config.py`.
  If you need to change the data or output locations, edit this file.
* Some scripts re-train models every time they are run. For faster
  experimentation, you can comment out training lines and load saved
  models instead.
* BERT training is **much faster on GPU**. On CPU it can take over an hour
  on the `all` dataset.

---

## 10. Contact

If you have questions about the code or would like to reuse it for
teaching/experiments, please contact the repository owner.

```
::contentReference[oaicite:0]{index=0}
```
