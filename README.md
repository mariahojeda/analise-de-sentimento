# analise-de-sentimento
Criado IA para análise de sentimento (textos em inglês) com integração a um BOT do Telegram - Atividade da Disciplina de IA



Fonte do banco de dados para treinamento: Kaggle.com
Dataset utilizado: Twitter Tweets Sentiment Dataset

Criação do ambiente virtual no terminal
conda create -n sentiment_analysis python=3.8

Ativação de ambiente virtual no terminal
conda activate sentiment_analysis

Bibliotecas instaladas dentro do ambiente virtual criado acima:
pip install python-telegram-bot==13.15
pip install numpy
pip install pandas
pip install nltk
pip install scikit-learn
pip install keras
pip install matplotlib

Discussão:
O botSentimento2.py foi desenvolvido com a intenção de aprimorar a capacidade de
resposta no que diz respeito à classificação de sentimentos.
Apesar da implementação do "early stopping", que contribuiu para uma discreta redução
na perda (figuras 2 e 6) durante o treinamento e validação do modelo, a acurácia obtida
permaneceu similar à versão anterior, o botSentimento.py.
Essa semelhança na acurácia (figuras 3 e 7) entre as duas versões pode indicar que,
embora o treinamento tenha sido refinado, a dificuldade em classificar corretamente
textos neutros ou negativos como positivos persistem, resultando em um retorno de
resultados equivalente (figuras 4 e 8) entre os dois códigos.
O número de épocas e o tamanho do lote são parâmetros críticos no treinamento de
modelos de aprendizado profundo, pois afetam diretamente a qualidade do treinamento.
No entanto, devido à falta de memória, tanto em execuções locais quanto em ambientes
Conda, não foi possível aumentar esses valores significativamente.
