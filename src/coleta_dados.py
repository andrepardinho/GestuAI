"""
coleta_dados.py — Etapa 1: Captura de landmarks e geração do dataset.csv

Estrutura pensada para ser ESCALÁVEL: adicionar ou remover quais landmarks
são gravados é feito em UM só lugar (LANDMARK_CONFIG), sem precisar mexer
na lógica de extração, nomeação de colunas ou gravação no CSV.

Partes propositalmente deixadas como TODO (decidir depois):
- Estratégia da classe "neutro"
- Critério de tempo/quantidade de frames por pose
- Lógica de variação (distância, inclinação, etc.) durante a captura
"""

import os
import csv
import cv2
import mediapipe as mp

# =============================================================================
# 1. CONFIGURAÇÃO DE LANDMARKS (ponto único de escalabilidade)
# =============================================================================
# Cada grupo aponta para um dos resultados do MediaPipe Holistic:
#   - "pose"       -> results.pose_landmarks       (33 landmarks no total)
#   - "face"       -> results.face_landmarks       (468 landmarks no total)
#   - "left_hand"  -> results.left_hand_landmarks  (21 landmarks no total)
#   - "right_hand" -> results.right_hand_landmarks (21 landmarks no total)
#
# "total" é o número TOTAL de landmarks que o MediaPipe retorna para aquele
# grupo (usado para saber o tamanho do preenchimento com zeros quando o
# grupo não for detectado). "indices" é a lista dos landmarks que você
# efetivamente quer gravar — pode ser um subconjunto.
#
# Para adicionar um landmark: inclua o índice na lista "indices" do grupo.
# Para remover: apague o índice da lista.
# Para adicionar um grupo novo (ex: pernas): copie o padrão de um bloco.

mp_holistic = mp.solutions.holistic
Pose = mp_holistic.PoseLandmark

LANDMARK_CONFIG = {
    "pose": {
        "total": 33,
        "indices": [
            Pose.NOSE.value,
            Pose.LEFT_EYE.value,
            Pose.RIGHT_EYE.value,
            Pose.MOUTH_LEFT.value,
            Pose.MOUTH_RIGHT.value,
            Pose.LEFT_SHOULDER.value,
            Pose.RIGHT_SHOULDER.value,
            Pose.LEFT_ELBOW.value,
            Pose.RIGHT_ELBOW.value,
            Pose.LEFT_WRIST.value,
            Pose.RIGHT_WRIST.value,
        ],
        # Se True, inclui a coluna "visibility" do MediaPipe além de x, y, z
        "include_visibility": True,
    },
    "left_hand": {
        "total": 21,
        "indices": list(range(21)),  # ex: list(range(21)) para gravar a mão inteira
        "include_visibility": True,
    },
    "right_hand": {
        "total": 21,
        "indices": list(range(21)),
        "include_visibility": True,
    },
    "face": {
        "total": 468,
        "indices": [],  # TODO: preencher com índices de olhos/boca se precisar
        "include_visibility": False,
    },
}

# Nome do atributo em `results` para cada grupo (mapeamento fixo do MediaPipe)
RESULT_ATTR = {
    "pose": "pose_landmarks",
    "face": "face_landmarks",
    "left_hand": "left_hand_landmarks",
    "right_hand": "right_hand_landmarks",
}

DATASET_PATH = "data/dataset.csv"


# =============================================================================
# 2. GERAÇÃO DE NOMES DE COLUNAS (derivada automaticamente da config acima)
# =============================================================================
def build_column_names(config: dict) -> list:
    """
    Gera os nomes das colunas do CSV a partir do LANDMARK_CONFIG.
    Como é derivado automaticamente, mudar a config já reflete no header
    sem precisar editar nada aqui.
    """
    columns = []
    for group_name, group_cfg in config.items():
        for idx in group_cfg["indices"]:
            columns.extend([
                f"{group_name}_{idx}_x",
                f"{group_name}_{idx}_y",
                f"{group_name}_{idx}_z",
            ])
            if group_cfg["include_visibility"]:
                columns.append(f"{group_name}_{idx}_v")
    columns.append("participant_id")  # Adiciona a coluna de identificação da pessoa
    columns.append("label")
    return columns


# =============================================================================
# 3. EXTRAÇÃO DE LANDMARKS (trata ausência de detecção preenchendo com zeros)
# =============================================================================
def extract_group(landmark_list, group_cfg: dict) -> list:
    """
    Extrai as coordenadas de um único grupo (pose/face/mãos), respeitando
    os índices definidos em group_cfg["indices"].

    Se landmark_list for None (ex: mão fora de quadro), preenche com zeros
    mantendo a MESMA quantidade de valores — isso é essencial para que o
    dataset tenha sempre a mesma estrutura de colunas (ver item 4 do
    planejamento: tratamento de dimensionalidade).
    """
    values = []
    n_coords = 4 if group_cfg["include_visibility"] else 3

    for idx in group_cfg["indices"]:
        if landmark_list is not None:
            lm = landmark_list.landmark[idx]
            point = [lm.x, lm.y, lm.z]
            if group_cfg["include_visibility"]:
                point.append(lm.visibility)
            values.extend(point)
        else:
            values.extend([0.0] * n_coords)

    return values


def extract_landmarks(results, config: dict) -> list:
    """
    Percorre todos os grupos configurados e monta um único array
    unidimensional com todas as coordenadas selecionadas, na mesma
    ordem usada por build_column_names().
    """
    row = []
    for group_name, group_cfg in config.items():
        attr_name = RESULT_ATTR[group_name]
        landmark_list = getattr(results, attr_name)
        row.extend(extract_group(landmark_list, group_cfg))
    return row


# =============================================================================
# 4. GRAVAÇÃO NO CSV (modo append, cria header se o arquivo não existir)
# =============================================================================
def save_row(row: list, participant_id: str, label: str, path: str = DATASET_PATH):
    file_exists = os.path.isfile(path)

    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(build_column_names(LANDMARK_CONFIG))

        writer.writerow(row + [participant_id, label])


# =============================================================================
# 5. LOOP PRINCIPAL DE CAPTURA
# =============================================================================
def main():
    # TODO: decidir se o nome da pose é pedido uma vez no início ou
    # pode ser trocado durante a execução (ex: teclas numéricas).
    pose_label = input("Nome da pose a ser gravada (ex: macaco_zen): ").strip()
    participant_id = input("ID/nome do participante (ex: Lules01): ").strip()

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        # TODO: definir aqui a lógica de quando começar/parar a gravação
        # (tempo corrido, número de frames, tecla de start/stop, etc.)
        # Por enquanto, o loop só mostra a webcam com os landmarks desenhados.
        recording = False

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Espelha horizontalmente a imagem para deixar a visualização
            # semelhante à de um espelho.
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # Desenho de referência visual (opcional, ajuda a validar a captura)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
            # Desenha a mão esquerda
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            
            # Desenha a mão direita (caso resolva ativar depois)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            
            if recording:
                row = extract_landmarks(results, LANDMARK_CONFIG)
                save_row(row, participant_id, pose_label)

            cv2.imshow("Coleta de Dados", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                recording = not recording  # liga/desliga gravação manualmente

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
