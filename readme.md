# GestuAI

![Python](https://img.shields.io/badge/python-3.12%20%7C%20<3.12-blue)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-orange)

## 📝 Descrição
O **GestuAI** é um sistema de visão computacional que realiza o reconhecimento de poses e gestos em tempo real através da webcam. O projeto serve para capturar a linguagem corporal do usuário e classificá-la automaticamente em diversas poses baseadas em figurinhas da internet (como "absolute cinema", "calabreso", "macaco reflexivo", entre outras).

**Tecnologias utilizadas:**
* **Python** como linguagem principal base.
* **MediaPipe** para a detecção e extração espacial dos landmarks corporais, faciais e das mãos.
* **OpenCV (`cv2`)** para captura de vídeo, espelhamento de imagem e interface visual na tela.
* **Scikit-Learn** para o treinamento e predição utilizando um modelo de Machine Learning do tipo Random Forest.
* **Pandas** e **Joblib** para manipulação dos datasets em CSV e salvamento/carregamento do modelo treinado.

## 💻 Pré-requisitos
* **Python:** É estritamente necessário utilizar a versão **3.12.x ou inferior**. Versões mais recentes do Python causarão incompatibilidade com a versão antiga do MediaPipe exigida pelo sistema.
* **Webcam:** Necessária para a captura de vídeo na inferência e na coleta de dados.
* Sistema operacional Windows, macOS ou Linux.

## 🛠️ Instalação
Recomenda-se o uso de um ambiente virtual (venv) para evitar conflitos de dependências.

```bash
# 1. Clone o repositório
git clone https://github.com/andrepardinho/GestuAI
cd GestuAI

# 2. Crie o ambiente virtual
python -m venv venv

# 3. Ative o ambiente virtual
# No Windows:
venv\Scripts\activate
# No Linux/MacOS:
source venv/bin/activate

# 4. Instale as dependências listadas
pip install -r requirements.txt
```

## 🚀 Como Usar
O projeto pode ser utilizado de duas formas principais: executando o modelo já treinado ou criando o seu próprio modelo do zero.

### 1. Testar o Reconhecimento ao Vivo
Se você já possui o arquivo `modelo_poses.pkl` na pasta `models`, basta rodar a aplicação principal:  
```bash
python app_circuito.py
```
Faça as poses na frente da câmera. Para encerrar o aplicativo, pressione a tecla `q` na janela do vídeo.

### 2. Criar e Treinar seu Próprio Modelo
Caso queira treinar novas figurinhas, o sistema possui um circuito de 3 etapas:
1. Coleta de Dados: Execute `python coleta_dados.py` para gravar seus movimentos e gerar arquivo de dados brutos.
Siga os avisos na tela para variar a posição. Pressione `s` para iniciar a contagem regressiva e gravar os frames da pose.   
2. Processamento e Normalização: Rode `python processar_dados.py` para padronizar o tamanho e posição das poses coletadas (centraliza o nariz e divide pela distância dos ombros).
3. Treinamento: Execute `python treinamento.py` para ensinar a Inteligência Artificial a reconhecer os dados que você gravou e gerar um novo modelo.

## 👩‍💻 Desenvolvedores

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/CauanValadao">
        <img src="https://avatars.githubusercontent.com/u/167363904?v=4" width="115px;" alt="Foto de Cauan"/><br />
        <sub><b>Cauan Valadão</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/andrepardinho">
        <img src="https://avatars.githubusercontent.com/u/153616098?v=4" width="115px;" alt="Foto de André"/><br />
        <sub><b>André Pardinho</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/svictoriasantana">
        <img src="https://avatars.githubusercontent.com/u/179859232?v=4" width="115px;" alt="Foto de Victória"/><br />
        <sub><b>Victória Santana</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/guibongestab">
        <img src="https://avatars.githubusercontent.com/u/159674836?v=4" width="115px;" alt="Foto de Guilherme"/><br />
        <sub><b>Guilherme Bongestab</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Thaina-0">
        <img src="https://avatars.githubusercontent.com/u/257646291?v=4" width="115px;" alt="Foto de Thainá"/><br />
        <sub><b>Thainá Guimarães</b></sub>
      </a>
    </td>
  </tr>
</table>
