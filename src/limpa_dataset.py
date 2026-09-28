"""
limpar_dataset.py — Remove do dataset as linhas em que um membro ESSENCIAL
para a figurinha foi gravado zerado (o MediaPipe perdeu o membro naquele frame).

Exemplo: "gatinho_legal" depende da mão direita. Se em algum frame dessa
figurinha todas as colunas right_hand_* estão em 0, aquela linha é apagada.
Linhas de outras figurinhas, e linhas da mesma figurinha em que o membro foi
detectado, ficam exatamente como estavam.

Como funciona:
  - Um membro é considerado ZERADO quando TODAS as colunas x/y/z do grupo
    são exatamente 0 (é assim que coleta_dados.py grava a ausência).
  - As linhas mantidas são copiadas como texto, sem reformatar nenhum número.
  - Por padrão o resultado vai para um arquivo novo (OUTPUT_PATH), preservando
    o dataset_raw.csv original. Depois de conferir o relatório, rode
    processar_dados.py apontando para o arquivo limpo (ou troque OUTPUT_PATH
    para o mesmo caminho do INPUT_PATH se quiser sobrescrever).
"""

import os
import pandas as pd

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
INPUT_PATH = "data/dataset_raw.csv"
OUTPUT_PATH = "data/dataset_raw_limpo.csv"

# True = só mostra o relatório, não grava nenhum arquivo.
DRY_RUN = False

# Avisa se uma figurinha perder mais que isso (%) das linhas — pode indicar
# membro escolhido errado ou figurinha mal gravada.
ALERTA_PERCENTUAL = 30.0

# Membros aceitos = nomes dos grupos nas colunas do CSV:
#   "right_hand" -> colunas right_hand_*   "left_hand" -> left_hand_*
#   "face"       -> colunas face_*         "pose"      -> pose_*
#
# ATENÇÃO: a coleta espelha a imagem (cv2.flip) antes do MediaPipe, então
# "right_hand" nas colunas pode ser a mão ESQUERDA real da pessoa. Use o nome
# do grupo como aparece no CSV (o mesmo que aparece na tela ao gravar).
#
# Uma figurinha pode ter vários membros: a linha é apagada se QUALQUER um
# deles estiver zerado.
ESSENCIAIS = {
    "crianca_chocada": ["right_hand"],
    "dedo_apontado": ["left_hand"],
    "gatinho_hang_loose": ["right_hand"],
    "edward_nojo": ["left_hand"],
    "macaco_reflexivo": ["left_hand"],
    "nao_grita": ["right_hand"],
    "gatinho_legal": ["left_hand"],
    # "gatinho_hang_loose": ["right_hand"],
    # "dedo_apontado": ["left_hand"],
    # "pensativo": ["right_hand", "face"],
}

GRUPOS_VALIDOS = ("pose", "left_hand", "right_hand", "face")


# =============================================================================
# FUNÇÕES
# =============================================================================
def colunas_do_grupo(colunas, grupo: str) -> list:
    """Colunas espaciais (x/y/z) do grupo. Visibility (_v) não conta."""
    return [c for c in colunas
            if c.startswith(f"{grupo}_") and c.endswith(("_x", "_y", "_z"))]


def validar_config(df: pd.DataFrame) -> None:
    if "label" not in df.columns:
        raise ValueError("O CSV não tem a coluna 'label'.")

    for figurinha, membros in ESSENCIAIS.items():
        for membro in membros:
            if membro not in GRUPOS_VALIDOS:
                raise ValueError(
                    f"Membro '{membro}' (figurinha '{figurinha}') inválido. "
                    f"Use um de: {GRUPOS_VALIDOS}"
                )
            if not colunas_do_grupo(df.columns, membro):
                raise ValueError(
                    f"Nenhuma coluna de '{membro}' encontrada no CSV "
                    f"(figurinha '{figurinha}')."
                )


def calcular_mascaras_zerado(df: pd.DataFrame) -> dict:
    """Para cada grupo usado na config: Series booleana (True = grupo zerado)."""
    grupos_usados = {m for membros in ESSENCIAIS.values() for m in membros}
    mascaras = {}
    for grupo in grupos_usados:
        cols = colunas_do_grupo(df.columns, grupo)
        valores = df[cols].astype(float)
        mascaras[grupo] = (valores == 0.0).all(axis=1)
    return mascaras


def limpar():
    # dtype=str: lê tudo como texto para que as linhas mantidas sejam
    # gravadas exatamente como estavam (nenhum número é reformatado).
    df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    validar_config(df)

    mascaras = calcular_mascaras_zerado(df)
    remover = pd.Series(False, index=df.index)
    relatorio = []

    for figurinha, membros in ESSENCIAIS.items():
        da_figurinha = df["label"] == figurinha
        total = int(da_figurinha.sum())

        if total == 0:
            print(f"[AVISO] Figurinha '{figurinha}' não existe no dataset.")
            continue

        remover_fig = pd.Series(False, index=df.index)
        por_membro = {}
        for membro in membros:
            m = da_figurinha & mascaras[membro]
            por_membro[membro] = int(m.sum())
            remover_fig |= m

        remover |= remover_fig
        apagadas = int(remover_fig.sum())
        relatorio.append((figurinha, total, apagadas, por_membro))

    # ---------------------------- RELATÓRIO ----------------------------------
    print("\n" + "=" * 68)
    print(f"RELATÓRIO DE LIMPEZA  ({INPUT_PATH})")
    print("=" * 68)

    for figurinha, total, apagadas, por_membro in relatorio:
        pct = 100.0 * apagadas / total
        detalhe = ", ".join(f"{m}: {n}" for m, n in por_membro.items())
        print(f"\n{figurinha}")
        print(f"  linhas antes : {total}")
        print(f"  apagadas     : {apagadas} ({pct:.1f}%)   [{detalhe}]")
        print(f"  linhas depois: {total - apagadas}")
        if pct > ALERTA_PERCENTUAL:
            print(f"  [ALERTA] mais de {ALERTA_PERCENTUAL:.0f}% removido — "
                  f"confira se o membro escolhido está certo.")
        if total - apagadas == 0:
            print("  [ALERTA] a figurinha ficou SEM nenhuma linha.")

    total_apagadas = int(remover.sum())
    print("\n" + "-" * 68)
    print(f"Total no dataset: {len(df)} | apagadas: {total_apagadas} | "
          f"restantes: {len(df) - total_apagadas}")
    print("Figurinhas fora da configuração não foram alteradas.")

    # ------------------------------ GRAVAÇÃO ---------------------------------
    if DRY_RUN:
        print("\n[DRY_RUN] Nenhum arquivo foi gravado.")
        return

    diretorio = os.path.dirname(OUTPUT_PATH)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)

    df[~remover].to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Dataset limpo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    limpar()