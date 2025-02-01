import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline
)

# ** Cihaz Ayarı (MPS/CPU)**
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ** Kullanılacak Modeller**
MODELS = {
    "BERT": "textattack/bert-base-uncased-SST-2",  # Metin sınıflandırma
    "BART": "facebook/bart-large-cnn",  # Özetleme
    "DeBERTa": "microsoft/deberta-large-mnli",  # Zero-shot sınıflandırma
    "T5": "t5-small",  # Metinden etik öneriler üretme
    "ALBERT": "albert-base-v2",  # Metin sınıflandırma
    "BERT_QA": "deepset/bert-base-cased-squad2"  # BERT Soru-Cevaplama
}

# ** Veri Setini Yükleme Fonksiyonu**
def load_dataset(json_path):
    print(" Veri seti yükleniyor...")
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            texts = data.get("text", [])  
        elif isinstance(data, list):
            texts = [entry.get("text", "") for entry in data]
        else:
            print(" Veri formatı hatalı!")
            return None

        print(f" Veri başarıyla yüklendi! Toplam {len(texts)} kayıt var.")
        return texts
    except Exception as e:
        print(f" Veri yükleme hatası: {e}")
        return None

# ** Model Yükleme Fonksiyonları**
def load_model(model_name):
    print(f" {model_name} modeli yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[model_name]).to(device)
    return tokenizer, model

def load_summarization_model():
    tokenizer = AutoTokenizer.from_pretrained(MODELS["BART"])
    model = AutoModelForSeq2SeqLM.from_pretrained(MODELS["BART"]).to(device)
    return tokenizer, model

# ** Metin Sınıflandırma (BERT ve ALBERT)**
def classify_text(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Etik" if prediction == 1 else "Etik Değil"

# ** Metin Özetleme (BART)**
def summarize_text(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], max_length=100)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ** Zero-Shot Etik Kategorilere Ayırma (DeBERTa)**
def zero_shot_classification(text):
    classifier = pipeline("zero-shot-classification", model=MODELS["DeBERTa"])
    labels = ["etik", "etik değil", "tarafsız", "yanıltıcı"]
    return classifier(text, candidate_labels=labels)["labels"][0]

# ** T5 ile Etik Öneriler Üretme**
def generate_suggestions(text):
    tokenizer = AutoTokenizer.from_pretrained(MODELS["T5"])
    model = AutoModelForSeq2SeqLM.from_pretrained(MODELS["T5"]).to(device)
    inputs = tokenizer(f"Etik öneri üret: {text}", return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], max_length=50)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ** BERT ile Soru-Cevap**
def answer_question(question, context):
    tokenizer = AutoTokenizer.from_pretrained(MODELS["BERT_QA"])
    model = AutoModelForQuestionAnswering.from_pretrained(MODELS["BERT_QA"]).to(device)
    
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)
    
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1])
    )
    
    return answer if answer.strip() else "Cevap bulunamadı"

# **Grafik Üretme Fonksiyonları**
def plot_bert_results(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="BERT_Result", data=df, palette="pastel")
    plt.title("BERT Modeli ile Etiket Dağılımı")
    plt.savefig("/Users/bilge/Downloads/bert_result_distribution.png")

def plot_summary_length(df):
    df["summary_length"] = df["BART_Summary"].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(6,4))
    sns.histplot(df["summary_length"], bins=10, kde=True, color="blue")
    plt.title("BART Özet Uzunluk Dağılımı")
    plt.savefig("/Users/bilge/Downloads/bart_summary_length.png")

def generate_graphs(df):
    plot_bert_results(df)
    plot_summary_length(df)

# **Ana Fonksiyon**
def main():
    DATASET_PATH = "/Users/bilge/Downloads/oasst1_data.json"

    texts = load_dataset(DATASET_PATH)
    if texts is None:
        return

    bert_tokenizer, bert_model = load_model("BERT")
    bart_tokenizer, bart_model = load_summarization_model()

    results = []
    for i, text in enumerate(texts[:50]):
        print(f"{i+1}. veri işleniyor...")
        bert_result = classify_text(bert_tokenizer, bert_model, text)
        bart_result = summarize_text(bart_tokenizer, bart_model, text)
        results.append({"text": text, "BERT_Result": bert_result, "BART_Summary": bart_result})

    df = pd.DataFrame(results)
    results_path = "/Users/bilge/Downloads/ethics_analysis_results.csv"
    df.to_csv(results_path, index=False)

    generate_graphs(df)
    print("Sonuçlar ve grafikler kaydedildi.")

if __name__ == "__main__":
    main()
