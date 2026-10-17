from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import json
import uuid
import logging
import os
import base64
import numpy as np
import cv2
from pathlib import Path
from groq import Groq
from supabase import create_client, Client  # NOVO

# O Render vai injetar essas variáveis. Se rodar local, ele usa o valor padrão.

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "SUA_NOVA_CHAVE_REAL_AQUI")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "SUA_NOVA_CHAVE_REAL_AQUI")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "SUA_NOVA_CHAVE_REAL_AQUI")

client_groq = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Assistente Educacional Web")
BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. CONFIGURAÇÕES (GROQ + SUPABASE)
# ==========================================
client_groq = Groq(api_key=GROQ_API_KEY)

# COLE AQUI SUA URL E CHAVE ANON DO SUPABASE
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logger.info("✅ Groq e Supabase inicializados.")

# ==========================================
# 2. OPENCV SFACE
# ==========================================
try:
    detector = cv2.FaceDetectorYN_create(
        str(BASE_DIR / "modelos" / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.9, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF_create(
        str(BASE_DIR / "modelos" / "face_recognition_sface_2021dec.onnx"), ""
    )
except Exception as e:
    logger.error(f"❌ Erro OpenCV: {e}")
    detector, recognizer = None, None

rostos_encodings = []
ids_conhecidos = []

def carregar_rostos_servidor():
    """Carrega rostos diretamente do Storage do Supabase"""
    global rostos_encodings, ids_conhecidos
    try:
        # Lista todos os arquivos no bucket 'rostos'
        files = supabase.storage.from_('rostos').list()
        logger.info(f" Encontrados {len(files)} arquivos no bucket 'rostos'.")
        
        for file in files:
            file_name = file['name']
            if not file_name.endswith(('.jpg', '.png')): continue
            
            id_aluno = os.path.splitext(file_name)[0]
            
            # Baixa a imagem do Supabase
            img_bytes = supabase.storage.from_('rostos').download(file_name)
            nparr = np.frombuffer(img_bytes, np.uint8)
            imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if imagem is not None:
                h, w = imagem.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(imagem)
                if faces is not None and len(faces) > 0:
                    aligned = recognizer.alignCrop(imagem, faces[0])
                    feature = recognizer.feature(aligned)
                    rostos_encodings.append(feature)
                    ids_conhecidos.append(id_aluno)
                    
        logger.info(f"✅ {len(rostos_encodings)} rosto(s) carregados da nuvem.")
    except Exception as e:
        logger.error(f"Erro ao carregar rostos do Supabase: {e}")

carregar_rostos_servidor()

def comparar_rosto_web(feature_atual, tolerancia=0.4):
    if len(rostos_encodings) == 0: return None
    similaridades = []
    for i, feat in enumerate(rostos_encodings):
        sim = recognizer.match(feature_atual, feat, cv2.FaceRecognizerSF_FR_COSINE)
        similaridades.append((sim, ids_conhecidos[i]))
    similaridades.sort(reverse=True)
    maior_sim, id_mais_parecido = similaridades[0]
    if maior_sim > tolerancia:
        return id_mais_parecido
    return None

# ==========================================
# 3. BANCO DE DADOS (SUPABASE)
# ==========================================
CATEGORIAS_OFICIAIS = ["MATEMATICA", "QUIMICA", "FISICA", "BIOLOGIA", "HISTORIA", "GEOGRAFIA", "PORTUGUES", "INFORMATICA", "OUTROS"]

def salvar_pergunta_web(id_anonimo, ano, categoria, tema, pergunta, resposta):
    try:
        supabase.table('perguntas').insert({
            'id_aluno_anonimo': id_anonimo,
            'ano_escolar': ano,
            'categoria': categoria,
            'tema': tema,
            'pergunta': pergunta,
            'resposta': resposta
        }).execute()
    except Exception as e: logger.error(f"Erro ao salvar pergunta: {e}")

# ==========================================
# 4. ROTAS
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_file = BASE_DIR / "index.html"
    if not html_file.exists(): return HTMLResponse(content="<h1>Erro: index.html não encontrado</h1>", status_code=404)
    with open(html_file, "r", encoding="utf-8") as f: return f.read()

@app.post("/reconhecer")
async def reconhecer_rosto(request: Request):
    if detector is None or recognizer is None:
        return {"sucesso": False, "erro": "Modelos não carregados."}
    try:
        data = await request.json()
        image_b64 = data.get("imagem", "")
        if "," in image_b64: image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if imagem is None: return {"sucesso": False, "erro": "Imagem inválida."}
            
        h, w = imagem.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(imagem)
        if faces is None or len(faces) == 0:
            return {"sucesso": False, "erro": "Nenhum rosto detectado."}
            
        aligned = recognizer.alignCrop(imagem, faces[0])
        feature = recognizer.feature(aligned)
        id_detectado = comparar_rosto_web(feature)
        
        if id_detectado:
            # Busca no Supabase
            response = supabase.table('alunos').select('nome, ano_escolar').eq('id_aluno', id_detectado).execute()
            if response.data:
                nome_aluno = response.data[0]['nome']
                ano_escolar = response.data[0]['ano_escolar']
                logger.info(f"✅ Rosto reconhecido: {nome_aluno} ({ano_escolar})")
                return {"sucesso": True, "id": id_detectado, "nome": nome_aluno, "ano": ano_escolar}
            else:
                return {"sucesso": False, "erro": "Aluno encontrado no rosto, mas não no banco."}
        else:
            return {"sucesso": False, "erro": "Rosto não reconhecido."}
    except Exception as e:
        logger.error(f"Erro no reconhecimento: {e}")
        return {"sucesso": False, "erro": str(e)}

@app.post("/cadastrar")
async def cadastrar_rosto(request: Request):
    try:
        data = await request.json()
        image_b64 = data.get("imagem", "")
        nome = data.get("nome", "").strip()
        ano = data.get("ano", "").strip()

        if not nome or not ano:
            return {"sucesso": False, "erro": "Nome e ano são obrigatórios."}

        if "," in image_b64: image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if imagem is None: return {"sucesso": False, "erro": "Imagem inválida."}

        h, w = imagem.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(imagem)
        if faces is None or len(faces) == 0:
            return {"sucesso": False, "erro": "Nenhum rosto detectado."}

        id_aluno = str(uuid.uuid4())[:12]
        
        # 1. Salva a imagem no Storage do Supabase
        supabase.storage.from_('rostos').upload(f"{id_aluno}.jpg", img_bytes, {'content-type': 'image/jpeg'})
        
        # 2. Salva os dados no Banco
        supabase.table('alunos').insert({
            'id_aluno': id_aluno,
            'nome': nome,
            'ano_escolar': ano
        }).execute()

        # 3. Adiciona na memória RAM do servidor
        aligned = recognizer.alignCrop(imagem, faces[0])
        feature = recognizer.feature(aligned)
        rostos_encodings.append(feature)
        ids_conhecidos.append(id_aluno)

        logger.info(f"✅ Novo aluno cadastrado na nuvem: {nome} ({ano})")
        return {"sucesso": True, "mensagem": f"Cadastro realizado com sucesso, {nome}!"}

    except Exception as e:
        logger.error(f"Erro no cadastro: {e}")
        return {"sucesso": False, "erro": str(e)}

@app.post("/perguntar")
async def processar_pergunta(request: Request):
    try:
        data = await request.json()
        pergunta = data.get("pergunta", "").strip()
        ano_escolar = data.get("ano", "Visitante")
        id_sessao = data.get("id_sessao", str(uuid.uuid4())[:12])
        if not pergunta: return {"erro": "Pergunta vazia"}

        chat = [{"role": "system", "content": f"Você é um assistente educacional. Responda SEMPRE em JSON: {{\"resposta\": \"max 3 frases\", \"categoria\": \"MATEMATICA\", \"tema\": \"2-4 palavras\"}}. Categorias: {CATEGORIAS_OFICIAIS}."}]
        chat.append({"role": "user", "content": pergunta})

        texto_resposta = "Desculpe, tive um erro."
        categoria = "OUTROS"; tema = ""

        try:
            chat_completion = client_groq.chat.completions.create(
                messages=chat, model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}, temperature=0.5, max_tokens=1024, timeout=30
            )
            raw_text = chat_completion.choices[0].message.content.strip()
            if raw_text.startswith("```json"): raw_text = raw_text[7:]
            if raw_text.endswith("```"): raw_text = raw_text[:-3]
            dados = json.loads(raw_text)
            texto_resposta = dados.get("resposta", texto_resposta)
            categoria_raw = dados.get("categoria", "OUTROS").upper()
            tema = dados.get("tema", "")
            if categoria_raw in CATEGORIAS_OFICIAIS: categoria = categoria_raw
        except Exception as e:
            logger.error(f"Erro Groq: {e}")
            categoria = "ERRO"; texto_resposta = "Erro de conexão com a IA."

        if categoria != "ERRO": salvar_pergunta_web(id_sessao, ano_escolar, categoria, tema, pergunta, texto_resposta)
        return {"resposta": texto_resposta, "categoria": categoria, "tema": tema}
    except Exception as e: return {"erro": str(e)}

# if __name__ == "__main__":
    # import uvicorn 
    # from pyngrok import ngrok
    # ngrok.set_auth_token("")
    # public_url = ngrok.connect(8000)
    # logger.info(f"🚀 Servidor rodando em http://localhost:8000")
    # uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == "__main__":
    import uvicorn
    # O Render define a porta automaticamente. Se rodar local, usa 8000.
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Servidor iniciando na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)