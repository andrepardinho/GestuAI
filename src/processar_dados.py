"""
processar_dados.py — Geração do dataset normalizado.

Este script não implementa nenhuma regra matemática de normalização.

Toda a normalização está centralizada em normalizacao.py, garantindo
que o dataset usado no treinamento e os dados utilizados pelo
app_circuito.py durante a inferência sejam tratados pela mesma regra.
"""

import os
import pandas as pd

from normalizacao import aplicar_normalizacao_dataframe


RAW_PATH = "data/dataset_raw.csv"
NORM_PATH = "data/dataset_norm.csv"


def processar(raw_path: str = RAW_PATH,
              norm_path: str = NORM_PATH) -> pd.DataFrame:
    print(f"Lendo dataset bruto: {raw_path}")

    df = pd.read_csv(raw_path)

    print(f"Linhas carregadas: {len(df)}")

    df_normalizado = aplicar_normalizacao_dataframe(df)

    diretorio = os.path.dirname(norm_path)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)

    df_normalizado.to_csv(norm_path, index=False)

    print(f"[OK] {len(df_normalizado)} linhas processadas -> {norm_path}")

    return df_normalizado


if __name__ == "__main__":
    processar()