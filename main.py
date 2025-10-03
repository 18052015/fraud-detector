import pandas as pd
from fraud_detector import load_data, preprocess_data, train_model, predict_fraud

# Gerar dados fictícios para demonstração
def generate_fictitious_data(num_transactions=100):
    data = {
        'transaction_id': range(1, num_transactions + 1),
        'transaction_amount': [round(abs(x), 2) for x in pd.np.random.normal(loc=100, scale=50, size=num_transactions)],
        'transaction_type': pd.np.random.choice(['online', 'store', 'atm'], size=num_transactions),
        'timestamp': pd.to_datetime(pd.np.random.randint(1420070400, 1420070400 + 3600*24*30, num_transactions), unit='s')
    }
    df = pd.DataFrame(data)
    # Injetar algumas anomalias (valores muito altos)
    df.loc[5, 'transaction_amount'] = 5000.00
    df.loc[15, 'transaction_amount'] = 3000.00
    df.loc[25, 'transaction_amount'] = 4500.00
    return df

if __name__ == "__main__":
    print("Iniciando o Fraud Detector...")

    # 1. Gerar dados fictícios
    transactions_df = generate_fictitious_data(num_transactions=200)
    print("Dados fictícios gerados:")
    print(transactions_df.head())

    # 2. Pré-processar os dados
    processed_data = preprocess_data(transactions_df)
    print("\nDados pré-processados (amostra):")
    print(processed_data.head())

    # 3. Treinar o modelo
    model = train_model(processed_data)
    print("\nModelo de detecção de fraude treinado.")

    # 4. Prever fraudes
    fraud_predictions = predict_fraud(model, processed_data)

    # Adicionar previsões ao DataFrame original
    transactions_df['is_fraud'] = fraud_predictions
    transactions_df['is_fraud'] = transactions_df['is_fraud'].apply(lambda x: 'Fraud' if x == -1 else 'Normal')

    print("\nResultados da detecção de fraude:")
    print(transactions_df[transactions_df['is_fraud'] == 'Fraud'])

    print("\nDetecção de fraude concluída.")

