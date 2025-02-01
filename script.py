import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM, pipeline
)

# ** Cihaz Ayarı (MPS/CPU)**
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ** Model ve Tokenizer Yükleme**
MODEL_NAME_BERT = "textattack/bert-base-uncased-SST-2"
MODEL_NAME_BART = "facebook/bart-large-cnn"

bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BERT).to(device)

bart_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BART)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_BART).to(device)

# ** Veri Setini Yükleme**
def load_dataset(json_path):
    print(" Veri seti yükleniyor...")
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = [entry.get("text", "") for entry in data] if isinstance(data, list) else data.get("text", [])
    print(f" Veri başarıyla yüklendi! Toplam {len(texts)} kayıt var.")
    return texts

# ** Metin Sınıflandırma**
def classify_text(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction  # 1: Etik, 0: Etik Değil

# ** Metin Özetleme (BART)**
def summarize_text(text):
    inputs = bart_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        summary_ids = bart_model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ** Confusion Matrix ve Yanlış Tahmin Analizi**
def analyze_results(y_true, y_pred, texts):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Etik Değil", "Etik"], output_dict=True)

    print("\n **Model Performansı:**")
    print(classification_report(y_true, y_pred, target_names=["Etik Değil", "Etik"]))

    false_positives = [texts[i] for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1]
    false_negatives = [texts[i] for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0]

    # **Confusion Matrix Görselleştirme**
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Etik Değil", "Etik"], yticklabels=["Etik Değil", "Etik"])
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix")
    plt.savefig("/Users/bilge/Downloads/confusion_matrix.png")
    plt.show()

    return false_positives, false_negatives, report

# ** Feature Importance (Öznitelik Önemi) Analizi**
def feature_importance_analysis(texts):
    classifier = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

    importance_scores = []
    for text in texts[:20]:  
        tokenized_text = bert_tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        result = classifier(bert_tokenizer.decode(tokenized_text["input_ids"][0], skip_special_tokens=True))[0]
        importance_scores.append(result["score"])

    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(range(1, 21)), y=importance_scores, palette="coolwarm")
    plt.xlabel("Örnek No")
    plt.ylabel("Önem Skoru")
    plt.title("Feature Importance (Öznitelik Önem Skorları)")
    plt.savefig("/Users/bilge/Downloads/feature_importance.png")
    plt.show()

# ** Precision, Recall ve F1-score Bar Grafiği**
def plot_metrics(report):
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.iloc[:2]  

    plt.figure(figsize=(6, 4))
    sns.barplot(data=metrics_df[['precision', 'recall', 'f1-score']], palette="coolwarm")
    plt.title("Precision, Recall ve F1-score")
    plt.ylabel("Değer")
    plt.savefig("/Users/bilge/Downloads/bert_model_metrics.png")
    plt.show()

# ** BERT Modeli Sonuç Dağılımı**
def plot_bert_results(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="BERT_Result", data=df, palette="pastel")
    plt.title("BERT Modeli ile Etiket Dağılımı")
    plt.savefig("/Users/bilge/Downloads/bert_result_distribution.png")

# ** BART Özetleme Uzunluk Dağılımı**
def plot_summary_length(df):
    df["summary_length"] = df["BART_Summary"].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(6,4))
    sns.histplot(df["summary_length"], bins=10, kde=True, color="blue")
    plt.title("BART Özet Uzunluk Dağılımı")
    plt.savefig("/Users/bilge/Downloads/bart_summary_length.png")

# ** Ana Fonksiyon**
def main():
    DATASET_PATH = "/Users/bilge/Downloads/oasst1_data.json"
    texts = load_dataset(DATASET_PATH)

    y_true = [1 if "etik" in text.lower() else 0 for text in texts[:50]]
    y_pred = [classify_text(text) for text in texts[:50]]
    bart_summaries = [summarize_text(text) for text in texts[:50]]

    false_positives, false_negatives, report = analyze_results(y_true, y_pred, texts[:50])

    df_results = pd.DataFrame({
        "Text": texts[:50], 
        "BERT_Result": y_pred, 
        "BART_Summary": bart_summaries
    })

    df_fp = pd.DataFrame({"False Positive Örnekler": false_positives})
    df_fn = pd.DataFrame({"False Negative Örnekler": false_negatives})

    df_fp.to_csv("/Users/bilge/Downloads/false_positive_samples.csv", index=False)
    df_fn.to_csv("/Users/bilge/Downloads/false_negative_samples.csv", index=False)

    print(" Yanlış pozitif ve negatif örnekler kaydedildi.")

    feature_importance_analysis(texts[:50])

    plot_bert_results(df_results)
    plot_summary_length(df_results)
    plot_metrics(report)

if __name__ == "__main__":
    main()
