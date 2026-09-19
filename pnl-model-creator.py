import pandas as pd
import spacy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#etapa 1 : Carregar os dados
print("Carregando os dados...")
df = pd.read_csv("dataset_chamados - dataset_chamados.csv")

#etapa 2 pre processamento dos dados
#   # Substitua "dados.csv" pelo caminho do seu arquivo de dados
nlp = spacy.load("pt_core_news_sm") #modelo de pnl em português

def preprocess_text(text):
    doc = nlp(text) # processamento do texto com o modelo de PNL
    return " ".join(
            [token.lemma_.lower() for token in doc if not token.is_punct]
            #constroi uma lista em python de acordo com algumas operações
            # token.lemma_ -> retorna a forma lematizada do token
            # infinitivo, e substantivo no singular, etc
            #.lower() -> converte o token para minúsculo
            # for token in doc -> itera sobre cada token no objeto doc
            # return " ".join(...) -> junta todos os tokens processados em uma única string, separando-os por espaços

    )
print("Pré-processando os dados..., pode levar alguns segundos..." \
"")
df["texto_processado"] = df["texto"].apply(preprocess_text)
# aplicando a função preprocess_text a cada linha da coluna "texto" do DataFrame df, criando uma nova coluna "texto_processado" com os textos processados

#etapa 3: Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    df["texto_processado"], df["categoria"], test_size=0.2, random_state=42)

#etapa 4 : criar e treinar pipeline de ml
model_pipeline = make_pipeline(
    TfidfVectorizer(), #vetorizador de texto
    MultinomialNB()    #classificador naive bayes
)
model_pipeline.fit(X_train, y_train)
#cria uma sequencia de execução: transforma o texto pre processado em vetore
# com o método TfidfVectorizer e em seguida treina o classificador MultinomialNB com os dados de treino
# ajusta o nosso vetorizador com os dados de treino.


#etapa 5: Avaliar o modelo
print("\nRelatório de previsão:")
predicoes = model_pipeline.predict(X_test)
print(classification_report(y_test, predicoes))
#utiliza o pipeline que está treinado para prever as categorias dos dados de teste e gera um relatório de classificação com métricas como precisão, recall e f1-score.
# e emite o relatório de classificação, que mostra o desempenho do modelo em termos de precisão, recall e f1-score para cada categoria.

#etap 6: salvar o modelo treinado
joblib.dump(model_pipeline, "modelo_chamados.pkl")
print("\nModelo treinado e salvo como 'modelo_chamados.pkl'.")