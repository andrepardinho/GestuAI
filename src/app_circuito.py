import cv2
import joblib
import mediapipe as mp

from collections import Counter, deque

from normalizacao import aplicar_normalizacao


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_PATH = "models/modelo_poses.pkl"
BUFFER_SIZE = 10
MIN_CONFIDENCE = 0.60


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

from landmarks import (
    LANDMARK_CONFIG, RESULT_ATTR, build_feature_column_names,
    build_column_names, extract_group, extract_landmarks,
)

FEATURE_COLUMNS = build_feature_column_names(LANDMARK_CONFIG)


def normalizar_landmarks(features, results):
    """
    Aplica a MESMA normalização usada no treinamento — centralizar no
    nariz (face landmark 1) e dividir pela distância entre os ombros
    (pose landmarks 11 e 12) — reaproveitando normalizacao.py, o módulo
    compartilhado com a coleta/processamento em lote. Se a matemática de
    normalização mudar um dia, muda-se só em normalizacao.py e os três
    scripts (coleta, processamento em lote, inferência) continuam em
    sincronia.
    """
    return aplicar_normalizacao(features, FEATURE_COLUMNS, results)


# ============================================================
# BUFFER DAS PREVISÕES
# ============================================================

def classe_mais_frequente(buffer):
    if len(buffer) == 0:
        return None

    contador = Counter(buffer)

    return contador.most_common(1)[0][0]


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("Carregando modelo...")
    modelo = joblib.load(MODEL_PATH)
    print("Modelo carregado.")
    print("Classes conhecidas:")
    print(modelo.classes_)
    buffer = deque(maxlen=BUFFER_SIZE)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não foi possível abrir a webcam.")
        return

    with mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Espelha a imagem
            frame = cv2.flip(frame, 1)

            # OpenCV utiliza BGR
            # MediaPipe utiliza RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detecta os landmarks
            results = holistic.process(frame_rgb)

            # ------------------------------------------------
            # Desenho para debug
            # ------------------------------------------------

            mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)

            # ------------------------------------------------
            # Só tenta prever se o corpo foi detectado
            # ------------------------------------------------

            if results.pose_landmarks and results.face_landmarks:

                features = normalizar_landmarks(extract_landmarks(results), results)

                # O sklearn espera:
                #
                # [
                #   [feature1, feature2, ...]
                # ]
                #
                # por isso usamos [features]

                probabilidades = modelo.predict_proba([features])[0]
                melhor_indice = probabilidades.argmax()
                classe = modelo.classes_[melhor_indice]
                confianca = probabilidades[melhor_indice]

                # --------------------------------------------
                # Só adiciona ao buffer se houver confiança
                # suficiente
                # --------------------------------------------

                if confianca >= MIN_CONFIDENCE:

                    buffer.append(classe)

                classe_estavel = classe_mais_frequente(buffer)

                # --------------------------------------------
                # Mostrar informações na tela
                # --------------------------------------------

                texto_classe = (f"Pose: {classe_estavel}" if classe_estavel
                                else "Pose: desconhecida")
                texto_confianca = f"Confianca: {confianca:.2f}"
                cv2.putText(frame, texto_classe, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, texto_confianca, (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ------------------------------------------------
            # Exibe webcam
            # ------------------------------------------------

            cv2.imshow("GestuAI - Circuito", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()