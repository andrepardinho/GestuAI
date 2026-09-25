"""
processar_dados.py — Etapa intermediária: lê o dataset BRUTO
(dataset_raw.csv, gerado por coleta_dados.py) e produz o dataset
NORMALIZADO (dataset_norm.csv), usado para treinar o modelo.

Por que separado do coleta_dados.py?
Você decidiu que coleta_dados.py grava só os dados crus do MediaPipe.
Isso significa que se um dia você mudar a matemática de normalização
(trocar a referência do nariz, mudar o fator de escala, normalizar de
outro jeito), basta rodar este script de novo em cima do dataset_raw.csv
já existente — sem levar ninguém de volta para a webcam.

CONSISTÊNCIA COM O APP AO VIVO (app_circuito.py):
Este script reimplementa em Pandas (vetorizado, rápido para milhares de
linhas de uma vez) EXATAMENTE a mesma matemática de
normalizacao.aplicar_normalizacao():
  1. Centraliza cada linha subtraindo a coordenada do nariz
     (face_1_x, face_1_y, face_1_z) de todas as colunas _x/_y/_z.
  2. Divide todas as colunas _x/_y/_z pela distância euclidiana 3D entre
     ombro esquerdo (pose_11) e ombro direito (pose_12).
Se você mudar normalizacao.py (a normalização ao vivo), replique a
mudança aqui também — os dois precisam ficar sempre em sincronia.
"""

import pandas as pd

RAW_PATH = "data/dataset_raw.csv"
NORM_PATH = "data/dataset_norm.csv"

# Referência de centralização: ponta do nariz (FaceMesh, landmark 1) —
# mesma referência usada em normalizacao.aplicar_normalizacao().
COL_NARIZ = ("face_1_x", "face_1_y", "face_1_z")

# Referência de escala: ombro esquerdo (pose 11) e ombro direito (pose 12)
COL_OMBRO_ESQ = ("pose_11_x", "pose_11_y", "pose_11_z")
COL_OMBRO_DIR = ("pose_12_x", "pose_12_y", "pose_12_z")

# Colunas que nunca entram na matemática espacial
COLUNAS_IGNORADAS = {"participant_id", "label"}


# =============================================================================
# 1. DETECÇÃO DE COLUNAS ESPACIAIS
# =============================================================================
def identificar_colunas_espaciais(df: pd.DataFrame) -> list:
    """
    Pega automaticamente todas as colunas _x, _y, _z do CSV (e ignora _v,
    participant_id, label). Como é automático, se você mudar quais
    landmarks são capturados em LANDMARK_CONFIG (coleta_dados.py), este
    script continua funcionando sem precisar editar nada aqui.
    """
    return [
        c for c in df.columns
        if c not in COLUNAS_IGNORADAS and c.endswith(("_x", "_y", "_z"))
    ]


def validar_colunas_referencia(df: pd.DataFrame) -> None:
    necessarias = [*COL_NARIZ, *COL_OMBRO_ESQ, *COL_OMBRO_DIR]
    faltando = [c for c in necessarias if c not in df.columns]
    if faltando:
        raise ValueError(
            f"Colunas de referência ausentes no CSV: {faltando}. "
            f"Confira se LANDMARK_CONFIG ainda captura o nariz "
            f"(face índice 1) e os ombros (pose índices 11 e 12)."
        )


# =============================================================================
# 2. NORMALIZAÇÃO DE POSIÇÃO (translação — centralizar no nariz)
# =============================================================================
def centralizar_no_nariz(df: pd.DataFrame, colunas_espaciais: list) -> pd.DataFrame:
    """
    Para cada linha, subtrai a coordenada do nariz daquela mesma linha de
    TODAS as colunas espaciais. Vetorizado: nenhum loop linha a linha.

    df[colunas_x].sub(df[nariz_x], axis=0) alinha pelo índice da linha e
    subtrai o valor escalar daquela linha de cada uma das colunas.
    """
    df = df.copy()
    nariz_x, nariz_y, nariz_z = COL_NARIZ

    colunas_x = [c for c in colunas_espaciais if c.endswith("_x")]
    colunas_y = [c for c in colunas_espaciais if c.endswith("_y")]
    colunas_z = [c for c in colunas_espaciais if c.endswith("_z")]

    df[colunas_x] = df[colunas_x].sub(df[nariz_x], axis=0)
    df[colunas_y] = df[colunas_y].sub(df[nariz_y], axis=0)
    df[colunas_z] = df[colunas_z].sub(df[nariz_z], axis=0)
    return df


# =============================================================================
# 3. NORMALIZAÇÃO DE ESCALA (dividir pela distância entre ombros)
# =============================================================================
def calcular_escala_ombros(df: pd.DataFrame) -> pd.Series:
    """Distância euclidiana 3D entre os ombros, para todas as linhas de uma vez."""
    dx = df[COL_OMBRO_ESQ[0]] - df[COL_OMBRO_DIR[0]]
    dy = df[COL_OMBRO_ESQ[1]] - df[COL_OMBRO_DIR[1]]
    dz = df[COL_OMBRO_ESQ[2]] - df[COL_OMBRO_DIR[2]]
    return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5


def normalizar_escala(df: pd.DataFrame, colunas_espaciais: list,
                       escala: pd.Series, limite_minimo: float = 1e-6) -> pd.DataFrame:
    """
    Divide todas as colunas espaciais pela escala, linha a linha.

    Linhas com escala inválida (ombros não detectados / distância ~0)
    NÃO são divididas — ficam só centralizadas, igual ao comportamento de
    normalizar_escala() em normalizacao.py ao vivo (evita dividir por um
    número perto de zero e distorcer a linha toda).
    """
    df = df.copy()
    escala_valida = escala.where(escala >= limite_minimo)  # NaN onde inválida

    n_invalidas = int(escala_valida.isna().sum())
    if n_invalidas > 0:
        print(f"[AVISO] {n_invalidas} linha(s) com distância de ombros "
              f"inválida — mantidas apenas centralizadas, sem normalização "
              f"de escala.")

    divisor = escala_valida.fillna(1.0)  # onde inválida, divide por 1 (não altera)
    df[colunas_espaciais] = df[colunas_espaciais].div(divisor, axis=0)
    return df


# =============================================================================
# 4. PIPELINE COMPLETO
# =============================================================================
def processar(raw_path: str = RAW_PATH, norm_path: str = NORM_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    validar_colunas_referencia(df)

    colunas_espaciais = identificar_colunas_espaciais(df)

    # Ordem importa: primeiro centraliza (translação), depois divide (escala) —
    # mesma ordem de normalizacao.aplicar_normalizacao() ao vivo.
    df = centralizar_no_nariz(df, colunas_espaciais)
    escala = calcular_escala_ombros(df)
    df = normalizar_escala(df, colunas_espaciais, escala)

    df.to_csv(norm_path, index=False)
    print(f"[OK] {len(df)} linhas processadas -> {norm_path}")
    return df


if __name__ == "__main__":
    processar()
