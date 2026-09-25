"""
normalizacao.py — Funções de normalização de landmarks.

Este módulo existe separado de coleta_dados.py de propósito: a MESMA
matemática usada aqui precisa ser aplicada, sem nenhuma diferença, dentro
de app_circuito.py no momento da inferência. Se a coleta normalizar de um
jeito e a inferência normalizar de outro (ou esquecer de normalizar), o
modelo treinado vai receber, em produção, dados numa "linguagem" diferente
da que ele aprendeu — e a classificação degrada silenciosamente.

Import este módulo nos dois scripts:
    from normalizacao import aplicar_normalizacao, calcular_distancia_ombros
"""

import math


# =============================================================================
# 1. NORMALIZAÇÃO DE POSIÇÃO (centralizar em relação ao nariz)
# =============================================================================
def centralizar_por_referencia(row: list, feature_columns: list,
                                ref_x: float, ref_y: float, ref_z: float = 0.0) -> list:
    """
    Subtrai (ref_x, ref_y, ref_z) de TODAS as colunas _x, _y, _z do row.

    Por que subtrair e não outra operação? Subtração transforma uma
    coordenada absoluta ("o pulso está em x=0.62 da tela") numa coordenada
    relativa ("o pulso está a 0.10 de distância do nariz"). Isso remove o
    efeito de ONDE a pessoa está parada na imagem — só sobra a FORMA da
    pose, que é o que realmente importa para diferenciar uma figurinha de
    outra.

    'feature_columns' precisa ser exatamente a lista de nomes de colunas
    que corresponde, posição a posição, aos valores de 'row' (sem
    participant_id/label — essas duas não fazem parte da matemática).
    """
    novo_row = list(row)
    for i, nome_coluna in enumerate(feature_columns):
        if nome_coluna.endswith("_x"):
            novo_row[i] = row[i] - ref_x
        elif nome_coluna.endswith("_y"):
            novo_row[i] = row[i] - ref_y
        elif nome_coluna.endswith("_z"):
            novo_row[i] = row[i] - ref_z
        # colunas "_v" (visibility) não são coordenadas espaciais, não mexemos
    return novo_row


# =============================================================================
# 2. NORMALIZAÇÃO DE ESCALA (dividir pela distância entre ombros)
# =============================================================================
def calcular_distancia_ombros(pose_landmarks) -> float | None:
    """
    Distância euclidiana entre ombro esquerdo (índice 11) e ombro direito
    (índice 12) nos landmarks de pose do MediaPipe. Usamos ombros porque:
      - quase sempre visíveis, mesmo com a pessoa de lado ou parcialmente
        fora de quadro;
      - variam pouco entre poses (a "largura do corpo" é relativamente
        estável), diferente de mãos/braços que se movem muito.
    Retorna None se a pose não foi detectada nesse frame.
    """
    if pose_landmarks is None:
        return None
    l = pose_landmarks.landmark[11]  # LEFT_SHOULDER
    r = pose_landmarks.landmark[12]  # RIGHT_SHOULDER
    return math.sqrt((l.x - r.x) ** 2 + (l.y - r.y) ** 2 + (l.z - r.z) ** 2)


def normalizar_escala(row: list, feature_columns: list, escala: float | None) -> list:
    """
    Divide todas as colunas _x, _y, _z por 'escala'. Isso faz uma pessoa
    grande/perto da câmera e uma pessoa pequena/longe da câmera produzirem
    aproximadamente os MESMOS números para a mesma pose.

    Se 'escala' vier None ou muito perto de zero (ombros não detectados,
    ou pessoa de costas), não normalizamos por escala nesse frame — melhor
    manter os valores centralizados sem dividir do que dividir por um
    número instável e distorcer tudo.
    """
    if escala is None or escala < 1e-6:
        return row

    novo_row = list(row)
    for i, nome_coluna in enumerate(feature_columns):
        if nome_coluna.endswith(("_x", "_y", "_z")):
            novo_row[i] = row[i] / escala
    return novo_row


# =============================================================================
# 3. FUNÇÃO ÚNICA DE ENTRADA (chame só esta, dos dois scripts)
# =============================================================================
def aplicar_normalizacao(row: list, feature_columns: list, results) -> list:
    """
    Aplica, nessa ordem, centralização + escala, usando o próprio
    resultado do MediaPipe (results) daquele frame para achar o nariz e a
    distância dos ombros.

    Uso em coleta_dados.py e em app_circuito.py deve ser IDÊNTICO:
        row = extract_landmarks(results, LANDMARK_CONFIG)
        row = aplicar_normalizacao(row, FEATURE_COLUMNS, results)
    """
    if results.pose_landmarks is None or results.face_landmarks is None:
        # Sem pose ou sem rosto detectados não há referência confiável
        # (o nariz vem do face mesh); devolve a linha como veio.
        return row

    nariz = results.face_landmarks.landmark[1]  # face.NOSE
    row = centralizar_por_referencia(row, feature_columns, nariz.x, nariz.y, nariz.z)

    escala = calcular_distancia_ombros(results.pose_landmarks)
    row = normalizar_escala(row, feature_columns, escala)

    return row