import os
import random
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image, ImageSequence

# Importando as funções e configurações diretamente do landmarks.py
from landmarks import (
    LANDMARK_CONFIG,
    build_feature_column_names,
    extract_landmarks,
)
from normalizacao import aplicar_normalizacao

# ============================================================
# CONFIGURAÇÕES DE CAMINHO E DIMENSÕES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Garante que acha a pasta raiz do projeto mesmo se executado dentro de src/
PROJECT_ROOT = (
    os.path.abspath(os.path.join(BASE_DIR, ".."))
    if os.path.basename(BASE_DIR) == "src"
    else BASE_DIR
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "modelo_poses.pkl")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

BUFFER_SIZE = 10
MIN_CONFIDENCE = 0.60
FRAMES_PARA_LIMPAR_BUFFER = 8

# Largura do painel lateral da figurinha
PAINEL_WIDTH = 360

GESTO_BRINCADEIRA = "clones"
NUM_CLONES = 8

# ============================================================
# MEDIAPIPE & FEATURES
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

FEATURE_COLUMNS = build_feature_column_names(LANDMARK_CONFIG)


def normalizar_landmarks(features, results):
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
# VISUAL E PAINEL LATERAL
# ============================================================

def criar_painel_placeholder(altura, largura, mensagem="Aguardando..."):
    """Cria um painel neutro quando ainda não detectou pose ou faltar imagem."""
    painel = np.zeros((altura, largura, 3), dtype=np.uint8)
    cv2.rectangle(painel, (0, 0), (largura - 1, altura - 1), (40, 40, 40), 2)
    cv2.putText(
        painel,
        mensagem,
        (largura // 2 - 110, altura // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (160, 160, 160),
        2,
        cv2.LINE_AA,
    )
    return painel


def carregar_figurinhas():
    """Carrega as imagens estáticas e todos os frames dos GIFs na memória."""
    cache = {}
    extensoes = [".jpg", ".png", ".jpeg", ".gif"]

    if not os.path.exists(ASSETS_DIR):
        return cache

    for arquivo in os.listdir(ASSETS_DIR):
        nome_sem_ext, ext = os.path.splitext(arquivo)
        if ext.lower() not in extensoes:
            continue

        caminho = os.path.join(ASSETS_DIR, arquivo)
        frames = []
        duracoes = []

        if ext.lower() == ".gif":
            with Image.open(caminho) as gif:
                duracao_padrao = gif.info.get("duration", 100)
                for frame_gif in ImageSequence.Iterator(gif):
                    duracao = frame_gif.info.get("duration", duracao_padrao)
                    rgba = frame_gif.convert("RGBA")
                    fundo = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
                    img = np.array(Image.alpha_composite(fundo, rgba).convert("RGB"))
                    frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    duracoes.append(max(1, duracao or 100) / 1000.0)
        else:
            img = cv2.imread(caminho)
            if img is not None:
                frames.append(img)
                duracoes.append(0.0)

        if frames:
            cache[arquivo] = {
                "nome": nome_sem_ext.lower(),
                "frames": frames,
                "duracoes": duracoes,
                "duracao_total": sum(duracoes),
            }

    return cache


def montar_painel_figurinha(img, altura, largura):
    """Centraliza a figurinha no painel sem deformar a imagem."""
    painel = np.zeros((altura, largura, 3), dtype=np.uint8)
    altura_img, largura_img = img.shape[:2]
    escala = min(largura / largura_img, altura / altura_img)
    nova_largura = max(1, min(largura, int(largura_img * escala)))
    nova_altura = max(1, min(altura, int(altura_img * escala)))
    img = cv2.resize(img, (nova_largura, nova_altura))

    x = (largura - nova_largura) // 2
    y = (altura - nova_altura) // 2
    painel[y:y + nova_altura, x:x + nova_largura] = img
    return painel


def obter_painel_figurinha(classe_nome, altura, largura, cache, inicio_animacao):
    """
    Busca a foto em memória correspondente ao gesto.
    Tenta busca exata e, se falhar, busca por correspondência parcial 
    (ex: classe 'cinema' encontra 'absolute_cinema.jpg').
    """
    if not classe_nome:
        return criar_painel_placeholder(altura, largura, "Aguardando Gesto...")

    extensoes = [".jpg", ".png", ".jpeg", ".gif"]
    classe_lower = classe_nome.lower()
    figurinha = None

    # 1. Tenta correspondência exata
    for ext in extensoes:
        figurinha = cache.get(f"{classe_nome}{ext}")
        if figurinha is not None:
            break

    # 2. Busca por correspondência parcial na pasta assets
    if figurinha is None:
        for dados in cache.values():
            nome_arq_lower = dados["nome"]
            # Exemplo: 'cinema' está dentro de 'absolute_cinema'
            if classe_lower in nome_arq_lower or nome_arq_lower in classe_lower:
                figurinha = dados
                break

    if figurinha is not None:
        indice_frame = 0
        if len(figurinha["frames"]) > 1:
            tempo = (time.monotonic() - inicio_animacao) % figurinha["duracao_total"]
            for indice, duracao in enumerate(figurinha["duracoes"]):
                if tempo < duracao:
                    indice_frame = indice
                    break
                tempo -= duracao

        return montar_painel_figurinha(
            figurinha["frames"][indice_frame], altura, largura
        )

    return criar_painel_placeholder(altura, largura, f"Foto: {classe_nome}")


def desenhar_hud(frame, classe_estavel, confianca, mostrar_landmarks):
    """Desenha uma barra superior transparente com texto destacado e instrução de teclas."""
    h, w, _ = frame.shape

    # Faixa superior semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Linha divisória verde
    cv2.line(frame, (0, 90), (w, 90), (0, 230, 150), 2)

    texto_pose = (
        f"POSE: {classe_estavel.upper()}"
        if classe_estavel
        else "POSE: DETECTANDO..."
    )
    cor_texto = (0, 255, 150) if classe_estavel else (180, 180, 180)

    # Sombra e texto principal da Pose
    cv2.putText(
        frame,
        texto_pose,
        (27, 47),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        texto_pose,
        (25, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        cor_texto,
        2,
        cv2.LINE_AA,
    )

    # Texto de Confiança (Sem ç para não buga no OpenCV)
    if classe_estavel and confianca:
        texto_conf = f"Confianca: {confianca * 100:.1f}%"
        cv2.putText(
            frame,
            texto_conf,
            (27, 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            texto_conf,
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Rodapé informando a tecla de atalho para ocultar/mostrar landmarks
    status_lm = "ON" if mostrar_landmarks else "OFF"
    cv2.putText(
        frame,
        f"[L] Landmarks: {status_lm}  |  [Q] Sair",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("Carregando modelo...")
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        return

    modelo = joblib.load(MODEL_PATH)
    print("Modelo carregado.")
    print("Classes conhecidas:", modelo.classes_)

    cache_figurinhas = carregar_figurinhas()
    frames_sem_confianca = 0
    classe_anterior = None
    inicio_animacao = time.monotonic()

    buffer = deque(maxlen=BUFFER_SIZE)
    cap = cv2.VideoCapture(0)

    window_name = "GestuAI - Circuito"

    cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )


    if not cap.isOpened():
        print("Não foi possível abrir a webcam.")
        return

    clones_ativos = False
    mostrar_landmarks = True  # Controle de exibição das linhas/pontos
    nomes_clones = [f"GestuAI - Clone {i}" for i in range(NUM_CLONES)]

    sorteios_config = {
        "gatinho_legal": ["gatinho_legal.jpg", "emoji_legal.gif"]
    }
    
    imagem_sorteada = None
    gesto_sorteio_atual = None

    with mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # Desenha os pontos na câmera apenas se a opção estiver ativada
            if mostrar_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )

            classe_estavel = None
            confianca = 0.0
            deteccao_confiavel = False

            if results.pose_landmarks and results.face_landmarks:
                features = normalizar_landmarks(
                    extract_landmarks(results, LANDMARK_CONFIG), results
                )
                probabilidades = modelo.predict_proba([features])[0]
                melhor_indice = probabilidades.argmax()
                classe = modelo.classes_[melhor_indice]
                confianca = probabilidades[melhor_indice]

                if confianca >= MIN_CONFIDENCE:
                    buffer.append(classe)
                    deteccao_confiavel = True

            if deteccao_confiavel:
                frames_sem_confianca = 0
            else:
                frames_sem_confianca += 1
                if frames_sem_confianca >= FRAMES_PARA_LIMPAR_BUFFER:
                    buffer.clear()

            classe_estavel = classe_mais_frequente(buffer)

            if classe_estavel != classe_anterior:
                classe_anterior = classe_estavel
                inicio_animacao = time.monotonic()

            # 1. HUD com informações na câmera
            desenhar_hud(frame, classe_estavel, confianca, mostrar_landmarks)

            if classe_estavel in sorteios_config:
                # Sorteia apenas se não houver imagem travada ou se mudou de gatilho
                if imagem_sorteada is None or gesto_sorteio_atual != classe_estavel:
                    imagem_sorteada = random.choice(sorteios_config[classe_estavel])
                    gesto_sorteio_atual = classe_estavel
                
                nome_para_buscar = imagem_sorteada
            else:
                # Reseta o estado do sorteio para gestos normais
                imagem_sorteada = None
                gesto_sorteio_atual = None
                nome_para_buscar = classe_estavel

            # 2. Figurinha em assets/ com o nome sorteado (ou o nome padrão da classe)
            painel_figurinha = obter_painel_figurinha(
                nome_para_buscar, frame.shape[0], PAINEL_WIDTH,
                cache_figurinhas, inicio_animacao
            )

            # 3. Junta câmera + figurinha lado a lado
            tela_composta = cv2.hconcat([frame, painel_figurinha])

            # Brincadeira dos clones
            if classe_estavel == GESTO_BRINCADEIRA:
                if not clones_ativos:
                    for nome in nomes_clones:
                        cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(nome, 400, 300)
                        pos_x = random.randint(0, 1500)
                        pos_y = random.randint(0, 700)
                        cv2.moveWindow(nome, pos_x, pos_y)
                    clones_ativos = True

                for nome in nomes_clones:
                    cv2.imshow(nome, frame)
            else:
                if clones_ativos:
                    for nome in nomes_clones:
                        cv2.destroyWindow(nome)
                    clones_ativos = False

            cv2.imshow(window_name, tela_composta)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l") or key == ord("L"):
                mostrar_landmarks = not mostrar_landmarks  # Alterna visibilidade

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
