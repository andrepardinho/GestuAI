import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

DATASET_PATH = "data/dataset_norm.csv"
MODEL_PATH = "models/modelo_poses.pkl"


def carregar_dados():
    df = pd.read_csv(DATASET_PATH)

    print("Dataset carregado:")
    print(df.shape)

    return df


def separar_dados(df):
    X = df.drop(columns=["participant_id", "label"])
    y = df["label"]
    participants = df["participant_id"]

    return X, y, participants


def dividir_por_participante(X, y, participants, participante_teste):
    test_mask = participants == participante_teste
    train_mask = ~test_mask

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test


def treinar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    return modelo


def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAcurácia:")
    print(f"{accuracy:.2%}")

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))


def salvar_modelo(modelo):
    os.makedirs("models", exist_ok=True)

    joblib.dump(modelo, MODEL_PATH)

    print(f"\nModelo salvo em: {MODEL_PATH}")


def main():
    df = carregar_dados()

    X, y, participants = separar_dados(df)

    print("\nClasses:")
    print(y.value_counts())

    print("\nParticipantes:")
    print(participants.value_counts())

    participante_teste = "enzo"

    X_train, X_test, y_train, y_test = dividir_por_participante(
        X,
        y,
        participants,
        participante_teste
    )

    print("\nTreino:")
    print(X_train.shape)

    print("\nTeste:")
    print(X_test.shape)

    modelo = treinar_modelo(X_train, y_train)

    avaliar_modelo(modelo, X_test, y_test)

    salvar_modelo(modelo)


if __name__ == "__main__":
    main()