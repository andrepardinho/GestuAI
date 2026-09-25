"""
normalizacao.py — Funções compartilhadas de normalização de landmarks.

Este módulo é a ÚNICA fonte de verdade da normalização do projeto.

Ele oferece duas formas de aplicar exatamente a mesma regra:

1. aplicar_normalizacao()
   Usada durante a inferência em tempo real no app_circuito.py.

2. aplicar_normalizacao_dataframe()
   Usada por processar_dados.py para normalizar dataset_raw.csv em lote.

Normalização aplicada:
    1. Centralização em relação ao nariz.
    2. Divisão pela distância entre os ombros.

Landmarks ausentes originalmente representados por (0, 0, 0)
continuam exatamente (0, 0, 0) após a normalização.

Colunas de visibility (_v) nunca são normalizadas.
"""

import math
import pandas as pd


# =============================================================================
# CONFIGURAÇÕES DAS REFERÊNCIAS
# =============================================================================

# FaceMesh landmark 1: ponta/região central do nariz
COL_NARIZ = ("face_1_x", "face_1_y", "face_1_z")

# MediaPipe Pose: 11 = ombro esquerdo, 12 = ombro direito
COL_OMBRO_ESQ = ("pose_11_x", "pose_11_y", "pose_11_z")
COL_OMBRO_DIR = ("pose_12_x", "pose_12_y", "pose_12_z")

COLUNAS_NAO_FEATURES = {"participant_id", "label"}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def eh_coluna_espacial(nome_coluna: str) -> bool:
    """
    Retorna True somente para coordenadas espaciais x, y ou z.

    Visibility (_v), participant_id e label não entram na normalização.
    """
    return (nome_coluna not in COLUNAS_NAO_FEATURES
            and nome_coluna.endswith(("_x", "_y", "_z")))


def obter_prefixo_landmark(nome_coluna: str) -> str:
    """
    Remove o sufixo _x, _y ou _z.

    Exemplo:
        left_hand_5_x -> left_hand_5
        pose_11_z     -> pose_11
    """
    return nome_coluna.rsplit("_", 1)[0]


def identificar_landmarks_ausentes(row: list, feature_columns: list) -> set:
    """
    Identifica landmarks cujo x, y e z originais são todos 0.

    Retorna os prefixos desses landmarks.

    Exemplo:
        left_hand_3_x = 0
        left_hand_3_y = 0
        left_hand_3_z = 0

    Resultado:
        {"left_hand_3"}

    É importante verificar o TRIO x/y/z, e não um valor isolado,
    pois uma coordenada individual igual a zero pode ser válida.
    """
    valores = {}
    for i, nome_coluna in enumerate(feature_columns):
        if not eh_coluna_espacial(nome_coluna):
            continue
        prefixo = obter_prefixo_landmark(nome_coluna)
        eixo = nome_coluna.rsplit("_", 1)[1]
        valores.setdefault(prefixo, {})[eixo] = row[i]

    ausentes = set()
    for prefixo, coords in valores.items():
        if (coords.get("x") == 0.0 and coords.get("y") == 0.0
                and coords.get("z") == 0.0):
            ausentes.add(prefixo)

    return ausentes


# =============================================================================
# NORMALIZAÇÃO DE UMA AMOSTRA
# Usada pelo app_circuito.py
# =============================================================================

def centralizar_por_referencia(row: list, feature_columns: list, ref_x: float,
                               ref_y: float, ref_z: float,
                               landmarks_ausentes: set) -> list:
    """
    Subtrai a posição do nariz das coordenadas espaciais.

    Landmarks ausentes permanecem em (0,0,0).
    Visibility não é alterada.
    """
    novo_row = list(row)
    for i, nome_coluna in enumerate(feature_columns):
        if not eh_coluna_espacial(nome_coluna):
            continue
        prefixo = obter_prefixo_landmark(nome_coluna)

        # Não transforma ausência em coordenada artificial.
        if prefixo in landmarks_ausentes:
            novo_row[i] = 0.0
            continue

        if nome_coluna.endswith("_x"):
            novo_row[i] = row[i] - ref_x
        elif nome_coluna.endswith("_y"):
            novo_row[i] = row[i] - ref_y
        elif nome_coluna.endswith("_z"):
            novo_row[i] = row[i] - ref_z

    return novo_row


def calcular_distancia_ombros(pose_landmarks) -> float | None:
    """
    Calcula a distância euclidiana 3D entre os ombros.

    Retorna None se os landmarks de pose não estiverem disponíveis.
    """
    if pose_landmarks is None:
        return None

    ombro_esq = pose_landmarks.landmark[11]
    ombro_dir = pose_landmarks.landmark[12]

    return math.sqrt((ombro_esq.x - ombro_dir.x) ** 2
                     + (ombro_esq.y - ombro_dir.y) ** 2
                     + (ombro_esq.z - ombro_dir.z) ** 2)


def normalizar_escala(row: list, feature_columns: list, escala: float | None,
                      landmarks_ausentes: set) -> list:
    """
    Divide coordenadas espaciais pela distância entre os ombros.

    Landmarks ausentes continuam em (0,0,0).
    """
    if escala is None or escala < 1e-6:
        return row

    novo_row = list(row)
    for i, nome_coluna in enumerate(feature_columns):
        if not eh_coluna_espacial(nome_coluna):
            continue
        prefixo = obter_prefixo_landmark(nome_coluna)

        if prefixo in landmarks_ausentes:
            novo_row[i] = 0.0
            continue

        novo_row[i] = row[i] / escala

    return novo_row


def aplicar_normalizacao(row: list, feature_columns: list, results) -> list:
    """
    Normalização usada durante a inferência ao vivo.

    Ordem:
        1. Detectar landmarks ausentes.
        2. Centralizar pelo nariz.
        3. Dividir pela distância entre os ombros.

    A detecção de ausências precisa ocorrer ANTES da normalização,
    enquanto os valores ainda são exatamente (0,0,0).
    """
    if results.pose_landmarks is None or results.face_landmarks is None:
        return row

    landmarks_ausentes = identificar_landmarks_ausentes(row, feature_columns)

    nariz = results.face_landmarks.landmark[1]

    row = centralizar_por_referencia(row, feature_columns, nariz.x, nariz.y,
                                     nariz.z, landmarks_ausentes)

    escala = calcular_distancia_ombros(results.pose_landmarks)

    row = normalizar_escala(row, feature_columns, escala, landmarks_ausentes)

    return row


# =============================================================================
# NORMALIZAÇÃO DE DATAFRAME
# Usada por processar_dados.py
# =============================================================================

def validar_colunas_referencia(df: pd.DataFrame) -> None:
    """
    Verifica se o dataset contém os landmarks necessários
    para realizar a normalização.
    """
    necessarias = [*COL_NARIZ, *COL_OMBRO_ESQ, *COL_OMBRO_DIR]
    faltando = [coluna for coluna in necessarias if coluna not in df.columns]

    if faltando:
        raise ValueError("Colunas necessárias para normalização ausentes: "
                         f"{faltando}")


def identificar_colunas_espaciais(df: pd.DataFrame) -> list:
    """
    Retorna somente as features espaciais x/y/z.
    """
    return [coluna for coluna in df.columns if eh_coluna_espacial(coluna)]


def identificar_mascaras_ausencia(df: pd.DataFrame,
                                  colunas_espaciais: list) -> dict:
    """
    Cria uma máscara booleana para cada landmark indicando quais
    linhas possuem exatamente (0,0,0).

    Exemplo:

        mascaras["left_hand_5"]

    contém True nas linhas em que aquele landmark estava ausente.
    """
    prefixos = {obter_prefixo_landmark(coluna) for coluna in colunas_espaciais}
    mascaras = {}

    for prefixo in prefixos:
        col_x, col_y, col_z = f"{prefixo}_x", f"{prefixo}_y", f"{prefixo}_z"

        if not all(coluna in df.columns for coluna in (col_x, col_y, col_z)):
            continue

        mascaras[prefixo] = ((df[col_x] == 0.0) & (df[col_y] == 0.0)
                             & (df[col_z] == 0.0))

    return mascaras


def calcular_escala_ombros_dataframe(df: pd.DataFrame) -> pd.Series:
    """
    Calcula a distância 3D entre os ombros para cada linha.
    """
    dx = df[COL_OMBRO_ESQ[0]] - df[COL_OMBRO_DIR[0]]
    dy = df[COL_OMBRO_ESQ[1]] - df[COL_OMBRO_DIR[1]]
    dz = df[COL_OMBRO_ESQ[2]] - df[COL_OMBRO_DIR[2]]

    return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5


def aplicar_normalizacao_dataframe(df: pd.DataFrame,
                                   limite_minimo: float = 1e-6) -> pd.DataFrame:
    """
    Aplica ao DataFrame a mesma normalização matemática usada
    por aplicar_normalizacao() durante a inferência.

    Regras:
        - subtrair nariz de x/y/z;
        - dividir x/y/z pela distância dos ombros;
        - não alterar visibility;
        - não alterar participant_id nem label;
        - preservar landmarks originalmente (0,0,0).
    """
    validar_colunas_referencia(df)

    resultado = df.copy()
    colunas_espaciais = identificar_colunas_espaciais(resultado)

    # IMPORTANTE: detectar zeros ANTES de qualquer transformação.
    mascaras_ausencia = identificar_mascaras_ausencia(resultado,
                                                      colunas_espaciais)

    # Escala calculada sobre coordenadas originais.
    # A distância não muda com a translação pelo nariz.
    escala = calcular_escala_ombros_dataframe(resultado)

    nariz_x, nariz_y, nariz_z = COL_NARIZ

    colunas_x = [c for c in colunas_espaciais if c.endswith("_x")]
    colunas_y = [c for c in colunas_espaciais if c.endswith("_y")]
    colunas_z = [c for c in colunas_espaciais if c.endswith("_z")]

    # -------------------------------------------------------------------------
    # 1. CENTRALIZAÇÃO
    # -------------------------------------------------------------------------

    resultado[colunas_x] = resultado[colunas_x].sub(resultado[nariz_x], axis=0)
    resultado[colunas_y] = resultado[colunas_y].sub(resultado[nariz_y], axis=0)
    resultado[colunas_z] = resultado[colunas_z].sub(resultado[nariz_z], axis=0)

    # -------------------------------------------------------------------------
    # 2. NORMALIZAÇÃO DE ESCALA
    # -------------------------------------------------------------------------

    escala_valida = escala.where(escala >= limite_minimo)
    n_invalidas = int(escala_valida.isna().sum())

    if n_invalidas > 0:
        print(f"[AVISO] {n_invalidas} linha(s) com distância "
              "de ombros inválida. Essas linhas serão apenas "
              "centralizadas.")

    divisor = escala_valida.fillna(1.0)
    resultado[colunas_espaciais] = (resultado[colunas_espaciais]
                                    .div(divisor, axis=0))

    # -------------------------------------------------------------------------
    # 3. RESTAURA LANDMARKS AUSENTES PARA (0,0,0)
    # -------------------------------------------------------------------------

    for prefixo, mascara in mascaras_ausencia.items():
        for eixo in ("x", "y", "z"):
            coluna = f"{prefixo}_{eixo}"
            if coluna in resultado.columns:
                resultado.loc[mascara, coluna] = 0.0

    return resultado