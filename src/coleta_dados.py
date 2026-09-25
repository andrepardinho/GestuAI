"""
coleta_dados.py — Etapa 1: Captura de landmarks BRUTOS e geração do
dataset_raw.csv

Este script grava exclusivamente coordenadas cruas do MediaPipe Holistic,
sem nenhuma normalização — a normalização (centralizar no nariz + dividir
pela distância dos ombros) fica em processar_dados.py, que lê
dataset_raw.csv e gera dataset_norm.csv em lote. Assim, a matemática de
normalização pode ser revista no futuro sem precisar regravar poses.

Estrutura pensada para ser ESCALÁVEL: adicionar ou remover quais landmarks
são gravados é feito em UM só lugar (LANDMARK_CONFIG), sem precisar mexer
na lógica de extração, nomeação de colunas ou gravação no CSV.

Recursos desta versão (ver comentários de cada etapa no corpo do arquivo):
  1. Controle automático de volume de captura (para sozinho perto de 1000 frames)
  2. Avisos em tela para induzir variação durante a gravação
  3. Fluxo explícito para gravar a classe "neutro"
  4. Contagem regressiva antes de iniciar a gravação
"""

import math
import os
import csv
import random
import time
import cv2
import mediapipe as mp

# =============================================================================
# 1. CONFIGURAÇÃO DE LANDMARKS (ponto único de escalabilidade)
# =============================================================================
mp_holistic = mp.solutions.holistic
Pose = mp_holistic.PoseLandmark

LANDMARK_CONFIG = {
    "pose": {
        "indices": [
            Pose.LEFT_SHOULDER.value,
            Pose.RIGHT_SHOULDER.value,
            Pose.LEFT_ELBOW.value,
            Pose.RIGHT_ELBOW.value,
            Pose.LEFT_WRIST.value,
            Pose.RIGHT_WRIST.value,
        ],
        "include_visibility": True
    },

    "left_hand": {
        "indices": list(range(21)),
        "include_visibility": False
    },

    "right_hand": {
        "indices": list(range(21)),
        "include_visibility": False
    },

    "face": {
        "indices": [
            1,          # nariz / referência
            61, 291,    # cantos da boca
            0, 17,      # lábios externos
            13, 14,     # lábios internos
            159, 145,   # olho
            386, 374    # outro olho
        ],
        "include_visibility": False
    },
}

RESULT_ATTR = {
    "pose": "pose_landmarks",
    "face": "face_landmarks",
    "left_hand": "left_hand_landmarks",
    "right_hand": "right_hand_landmarks",
}

DATASET_PATH = "data/dataset_raw.csv"

# --- Etapa 3: controle de volume de captura -------------------------------
#Quantos frames grava por clique do botão "s"
FRAMES_ALVO_POR_POSE = 200
# Quantos frames gravar por segundo no CSV (ex: 10 FPS)
FPS_GRAVACAO = 10 
INTERVALO_GRAVACAO = 1.0 / FPS_GRAVACAO

# --- Etapa 4: avisos de variação --------------------------------------------
# Trocados periodicamente em tela para lembrar o participante de variar.
AVISOS_VARIACAO = [
    "Mova-se um pouco: chegue mais perto ou afaste-se",
    "Incline levemente a cabeça",
    "Varie a altura dos braços",
    "Gire um pouco o corpo para o lado",
    "Mude sua posição no quadro",
]
INTERVALO_TROCA_AVISO_SEG = 2.5

# Nome reservado para a classe de "nenhuma pose" (Etapa 5)
LABEL_NEUTRO = "neutro"


# =============================================================================
# 2. GERAÇÃO DE NOMES DE COLUNAS (derivada automaticamente da config acima)
# =============================================================================
def build_column_names(config: dict) -> list:
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
    columns.append("participant_id")
    columns.append("label")
    return columns


def build_feature_column_names(config: dict) -> list:
    """
    Igual a build_column_names, mas SEM participant_id/label — é essa lista
    que a normalização usa, porque ela precisa saber, posição a posição em
    'row', quais entradas são x/y/z/v (participant_id e label não entram
    nessa matemática).
    """
    return build_column_names(config)[:-2]


# Não é mais usada para normalizar aqui (isso agora é em processar_dados.py),
# mas fica disponível caso você queira validar/gerar essa lista em algum
# teste ou script auxiliar.
FEATURE_COLUMNS = build_feature_column_names(LANDMARK_CONFIG)


# =============================================================================
# 3. EXTRAÇÃO DE LANDMARKS (trata ausência de detecção preenchendo com zeros)
# =============================================================================
def extract_group(landmark_list, group_cfg: dict) -> list:
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    pose_label = input(
        f"Nome da pose a ser gravada (ex: macaco_zen, ou '{LABEL_NEUTRO}'): "
    ).strip()
    participant_id = input("ID/nome do participante (ex: Lules01): ").strip()

    is_neutro = pose_label.lower() == LABEL_NEUTRO

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as holistic:

        recording = False
        frame_count = 0  # Etapa 3: quantos frames já foram gravados NESTA pose
        ultimo_tempo_gravado = 0
        is_counting_down = False     
        tempo_inicio_contagem = 0

        # Etapa 4: controle de qual aviso mostrar e quando trocar
        aviso_atual = random.choice(AVISOS_VARIACAO)
        proxima_troca_aviso = time.time() + INTERVALO_TROCA_AVISO_SEG

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            if not results.pose_landmarks or not results.face_landmarks:
                continue

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

            if results.face_landmarks:
                h, w, _ = frame.shape
                for idx in LANDMARK_CONFIG["face"]["indices"]:
                    lm = results.face_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            # --- Etapa 3 + 1 + 2: gravação com normalização e contador ------
            # --- Lógica de Contagem Regressiva ---
            if is_counting_down:
                tempo_restante = 5.0 - (time.time() - tempo_inicio_contagem)
                if tempo_restante <= 0:
                    is_counting_down = False
                    recording = True
                    frame_count = 0
                    ultimo_tempo_gravado = time.time()
                else:
                    # Desenha um texto gigante no centro da tela
                    segundos_inteiros = int(math.ceil(tempo_restante))
                    cv2.putText(frame, f"PREPARAR: {segundos_inteiros}", 
                                (frame.shape[1]//2 - 150, frame.shape[0]//2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 165, 255), 4)
            if recording:
                tempo_atual = time.time()
                if tempo_atual - ultimo_tempo_gravado >= INTERVALO_GRAVACAO:
                    row = extract_landmarks(results, LANDMARK_CONFIG)
                    save_row(row, participant_id, pose_label)
                    frame_count += 1
                    ultimo_tempo_gravado = tempo_atual


                    if frame_count >= FRAMES_ALVO_POR_POSE:
                        recording = False
                        print(
                            f"[OK] {frame_count} frames gravados para '{pose_label}'. "
                            f"Gravação parada automaticamente. Pressione 's' para "
                            f"iniciar outra rodada (ex: variando a posição)."
                        )
                        frame_count = 0

            # --- Etapa 4: troca o aviso de variação periodicamente ----------
            if time.time() >= proxima_troca_aviso:
                aviso_atual = random.choice(AVISOS_VARIACAO)
                proxima_troca_aviso = time.time() + INTERVALO_TROCA_AVISO_SEG

            # --- Textos na tela ----------------------------------------------
            status_texto = "GRAVANDO" if recording else "PARADO"
            status_cor = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"[{status_texto}] pose: {pose_label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_cor, 2)
            cv2.putText(frame, f"frames: {frame_count}/{FRAMES_ALVO_POR_POSE}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if recording:
                # Etapa 5: lembrete visual de que é a classe neutro,
                # para reforçar ao participante que não precisa "fazer pose"
                if is_neutro:
                    cv2.putText(frame, "Classe NEUTRO: fique parado ou se "
                                        "mova aleatoriamente, sem pose fixa",
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 165, 255), 2)
                else:
                    cv2.putText(frame, aviso_atual, (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.putText(frame, "[s] iniciar/pausar  [q] sair", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

            cv2.imshow("Coleta de Dados", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if recording:
                    # Se está a gravar, pausa imediatamente
                    recording = False
                elif is_counting_down:
                    # Se está a contar, cancela a contagem
                    is_counting_down = False
                else:
                    # Se está parado, inicia a contagem de 5 segundos
                    is_counting_down = True
                    tempo_inicio_contagem = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()