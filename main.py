import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import json

def preprocess_text(text: list, model, tokenizer):
    return model(**tokenizer(text, return_tensors='tf',
                                       padding=True, truncation=True))['last_hidden_state'][:, 0, :].numpy()

# Загрузка токенизатора и модели
bert_tokenizer = AutoTokenizer.from_pretrained("./rubert-tokenizer")
bert_model = TFAutoModel.from_pretrained("./rubert-model")

# Загрузка данных
with open("data.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

# Подготовка датасета
dataset = []
for q_a in data:
    embedding_question = preprocess_text([q_a["question"]], bert_model, bert_tokenizer)
    embedding_answer = preprocess_text([q_a["answer"]], bert_model, bert_tokenizer)
    dataset.append([embedding_question[0], embedding_answer[0]])

dataset = np.array(dataset)

# Загрузка обученной модели
model = tf.keras.models.load_model("FAQChatBotModel.keras")

def get_answer(question):
    if question.strip() == "":
        return "Вопрос не может быть пустым."

    # Преобразование вопроса в эмбеддинг
    embedding_question = preprocess_text([question], bert_model, bert_tokenizer)[0]

    # Сравнение с датасетом
    p = []
    for i in range(dataset.shape[0]):
        embedding_answer = dataset[i, 1]
        combined_embedding = np.concatenate([embedding_question, embedding_answer])
        prediction = model.predict(np.expand_dims(combined_embedding, axis=0), verbose=False)[0, 0]
        p.append([i, prediction])

    p = np.array(p)
    ans = np.argmax(p[:, 1])

    return "Ответ: " + data[ans]["answer"]

if __name__ == "__main__":
    print("Введите 'exit' для выхода.")
    while True:
        user_question = input("Ваш вопрос: ")
        if user_question.lower() == 'exit':
            break

        answer = get_answer(user_question)
        print(answer)
