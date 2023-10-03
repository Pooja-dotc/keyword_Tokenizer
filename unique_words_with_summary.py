from logging import _nameToLevel
import whisper
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import T5ForConditionalGeneration, T5Tokenizer
model = whisper.load_model("base")
options = whisper.DecodingOptions(fp16=False)
result = model.transcribe("C:\\Users\\kriti\\Downloads\\GadgetX.aac")
print(result["text"])

def abstractive_summarization(text):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

if __name__ == "__main__":
    input_text = result["text"]

    summary = abstractive_summarization(input_text)
    print("Original Text:")
    print(input_text)
    print("\nGenerated Summary:")
    print(summary)


nltk.download("stopwords")
nltk.download("punkt")

asr_model = whisper.load_model("base")
asr_options = whisper.DecodingOptions(fp16=False)

audio_file_path = "C:\\Users\\kriti\\Downloads\\GadgetX.aac"
transcription_result = asr_model.transcribe(audio_file_path)
transcribed_text = transcription_result["text"]

words = word_tokenize(transcribed_text)

stop_words = set(stopwords.words("english"))
filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]

fdist = nltk.FreqDist(filtered_words)

num_keywords = 10
keywords = [word for word, freq in fdist.most_common(num_keywords)]

print("Keywords used by the salesperson:")
for keyword in keywords:
    print(keyword)