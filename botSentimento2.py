import numpy as np # algebra linear
import pandas as pd # processamento de dados, leitura de arq CSV
import re # regular expression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle # para salvar modelo
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# bot telegram
#!pip install python-telegram-bot
#!pip install python-telegram-bot==13.15
#!pip install --upgrade python-telegram-bot
from telegram import Update
from telegram.ext import CallbackContext, Updater, CommandHandler, MessageHandler, Filters

data = pd.read_csv('Tweets.csv')
data.info()

# Dividir os dados em características (X) e rótulos (y)
X = data['text']
y = data['sentiment']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tamanho do conjunto de treinamento:", len(X_train))
print("Tamanho do conjunto de teste:", len(X_test))

stemmer = SnowballStemmer('english')

def preprocess_text(text):
    if isinstance(text, str):  # Verificar se o valor é uma string
        tokens = word_tokenize(text.lower())
        tokens = [stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        return ''
    
# Aplicar o pré-processamento aos conjuntos de treinamento e teste
X_train_preprocessed = X_train.apply(preprocess_text)
X_test_preprocessed = X_test.apply(preprocess_text)

# Inicializar o vetorizador
vetorizador = CountVectorizer(max_features=10000)

x_train = vetorizador.fit_transform(X_train_preprocessed)
x_test = vetorizador.transform(X_test_preprocessed)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train_encoded, num_classes)
y_test_onehot = to_categorical(y_test_encoded, num_classes)


# Criar o modelo sequencial
model = Sequential()

# Camada de entrada
model.add(Dense(units=128, activation='relu', input_dim=x_train.shape[1]))

# Camadas ocultas
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))

# Camada de saída
model.add(Dense(units=num_classes, activation='sigmoid'))

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitore a perda no conjunto de validação
    patience=5,  # Pare o treinamento após 5 épocas sem melhoria
    restore_best_weights=True  # Restaure os melhores pesos do modelo
)

# Converter matrizes CSR em matrizes NumPy
x_train = x_train.toarray()
x_test = x_test.toarray()

## APRENDIZADO ##
history = model.fit(
    x_train,
    y_train_onehot,
    epochs=10, 
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],  # Evitar overfitting
)

# Avaliação do modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(x_test, y_test_onehot)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot da perda
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot da acurácia
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


predito = model.predict(x_test)
print(predito)

# Real
print(y_test)

def avaliar_comentario(model, vetorizador, label_encoder, comentario):
    # Pré-processar o comentário
    comentario_preprocessado = preprocess_text(comentario)

    # Vetorizar o comentário
    comentario_vetorizado = vetorizador.transform([comentario_preprocessado])

    # Fazer a previsão do sentimento
    probabilidade_predita = model.predict(comentario_vetorizado)[0]

    # Decodificar a previsão em classe
    classe_predita = label_encoder.inverse_transform([np.argmax(probabilidade_predita)])[0]

    return classe_predita, probabilidade_predita

# Entrada do usuário
comentario_usuario = input("Digite um comentário: ")

# Avaliar o comentário usando o modelo treinado
classe_predita, probabilidade_predita = avaliar_comentario(model, vetorizador, label_encoder, comentario_usuario)

print("Sentimento predito:", classe_predita)
print("Probabilidades das classes:", probabilidade_predita)


## BOT TELEGRAM
# Configurações do bot
TOKEN = "6413276668:AAGrKUEARbnkLRSpOFod9JZNzroR5PCNTEk"

# Função para lidar com as mensagens do usuário
def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text
    classe_predita, _ = avaliar_comentario(model, vetorizador, label_encoder, user_message)
    response = f"Sentimento predito: {classe_predita}"
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)

def main():
    # Inicialização do bot
    updater = Updater(token='6413276668:AAGrKUEARbnkLRSpOFod9JZNzroR5PCNTEk', use_context=True)
    dispatcher = updater.dispatcher

    # Associar a função de tratamento de mensagens ao bot
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # Iniciar o bot
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()