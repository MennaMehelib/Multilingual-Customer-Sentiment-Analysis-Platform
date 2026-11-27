# Advanced NLP Model — XLM-RoBERTa

Part of the **Multilingual Customer Sentiment Analysis Platform**.
Fine-tunes `xlm-roberta-base` to classify reviews in Arabic, English and Chinese into 3 classes: `negative`, `neutral`, `positive`.

## Model

| Item | Value |
|---|---|
| Base model | `xlm-roberta-base` |
| Input text | `cleaned_text` column |
| Max length | 64 tokens |
| Epochs / batch size | 3 / 64 |
| Optimizer | AdamW, lr 2e-5, weight decay 0.01 |
| Scheduler | Linear warmup (10%) then linear decay |
| Precision | Mixed precision (fp16) |
| Model selection | Best epoch by validation macro F1 |

Data: `data/processed/splits` (train 90,000 / validation 13,500 / test 13,500), balanced across the 3 classes.

## Results (test set, macro average)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression + TF-IDF (best baseline) | 0.6221 | 0.6469 | 0.6221 | 0.6241 |
| **XLM-RoBERTa** | **0.7370** | **0.7362** | **0.7370** | **0.7364** |

Full results are in `results/`.

## Folder contents

```
Advanced NLP Model/
├── train.ipynb          # training and evaluation notebook (Google Colab, GPU)
├── predict.py           # run the model on new text
├── requirements.txt     # required libraries
├── README.md
├── results/
│   ├── metrics.json             # test metrics
│   ├── confusion_matrix.png     # test confusion matrix
│   ├── test_predictions.csv     # predictions for every test row (for error analysis)
│   └── sample_predictions.csv   # predictions on sample sentences
└── saved_model/         # model + tokenizer (not stored in the repo, see below)
```

## Model download

The model weights are about 1 GB, so they are not stored in this repository.

Download: https://drive.google.com/file/d/1YNEesY9qzzsh1jQUOsAhaWS9zy3SvvZk/view?usp=sharing

Unzip the files into a folder named `saved_model/` next to `predict.py`.

## Run

```bash
pip install -r requirements.txt
python predict.py "The service was excellent" "الخدمة ممتازة" "这个产品质量很好"
```

Output format: `label (confidence): text`

## Reproduce the training

1. Open `train.ipynb` in Google Colab and select a GPU runtime (T4).
2. Run all cells. The notebook clones the repository, reads the splits, trains the model and saves `saved_model/` and `results/`.
