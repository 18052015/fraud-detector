import pandas as pd
from sklearn.ensemble import IsolationForest

def load_data(filepath):
    """Carrega os dados de transações."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Pré-processa os dados para detecção de fraude."""
    # Exemplo simples: converter colunas categóricas em numéricas
    # Em um projeto real, isso seria mais complexo
    df['transaction_amount_usd'] = df['transaction_amount'] # Assumindo que a moeda é USD
    return df[['transaction_amount_usd']]

def train_model(data):
    """Treina um modelo de detecção de anomalias (fraude)."""
    model = IsolationForest(random_state=42)
    model.fit(data)
    return model

def predict_fraud(model, data):
    """Prevê transações fraudulentas."""
    # -1 para anomalias (fraude), 1 para transações normais
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    print("Módulo Fraud Detector carregado. Para usar, importe as funções.")
    # Exemplo de uso (requer um arquivo transactions.csv)
    # df = load_data('transactions.csv')
    # processed_data = preprocess_data(df)
    # model = train_model(processed_data)
    # fraud_predictions = predict_fraud(model, processed_data)
    # print(fraud_predictions)

