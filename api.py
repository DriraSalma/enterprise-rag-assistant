import os
import json
import torch  # Requis pour activer la détection automatique du GPU (CUDA / MPS)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse  # Import pour le streaming HTTP [3]
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

app = FastAPI(
    title="HTH-RAG API (ClinicSys) - Version Conversationnelle",
    description="API de streaming asynchrone local HTH-RAG avec mémoire de l'historique",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION DU SCHÉMA DE L'HISTORIQUE ---
class Message(BaseModel):
    role: str     # "user" ou "assistant"
    content: str

class RequeteRAG(BaseModel):
    question: str
    history: list[Message] = []  # Liste des messages d'historique (vide par défaut)
# --- CONFIGURATION MATÉRIELLE OPTIMISÉE (VRAM TRANSFER) ---
# On force l'Embedding et le Reranker sur le CPU (0 Mo de VRAM consommés par Python).
# Cela libère 100% de la mémoire de votre carte MX550 pour qu'Ollama accélère Mistral !
device_embedding = "cpu"
device_reranker = "cpu"

print("[+] Configuration VRAM optimisée : Encodage et Re-ranking sur le CPU.")
print("[+] Les 2 Go de VRAM de la MX550 sont entièrement libérés pour accélérer Ollama.")

client = QdrantClient(url="http://localhost:6333")
collection_name = "hth_rag_collection"

# Chargement stable des modèles sur le CPU
model = SentenceTransformer('BAAI/bge-m3', device=device_embedding)
reranker = CrossEncoder('BAAI/bge-reranker-base', device=device_reranker)
model_id = "mistral"  # Modèle Mistral de 7B local [2]
#model_id = "ministral-3:3b"
@app.post("/query")
async def interroger_rag(requete: RequeteRAG):
    query = requete.question
    if not query.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")
        
    try:
        # --- PHASE 1 : RECHERCHE DENSE (SÉMANTIQUE) ---
        query_vector = model.encode(query, convert_to_numpy=True).tolist()
        
        resultats_dense = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="child"))]),
            limit=8
        )
        
        # --- PHASE 2 : RECHERCHE SPARSE (LEXICALE) ---
        resultats_sparse, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="type", match=MatchValue(value="child")),
                    FieldCondition(key="description", match=MatchText(text=query))
                ]
            ),
            limit=8
        )
        
        # --- PHASE 3 : FUSION ET DÉDOUBLONNAGE ---
        candidats_uniques = {}
        for hit in resultats_dense.points:
            desc = hit.payload.get("description")
            if desc:
                candidats_uniques[desc] = hit.payload
            
        for point in resultats_sparse:
            desc = point.payload.get("description")
            if desc and desc not in candidats_uniques:
                candidats_uniques[desc] = point.payload
                
        liste_candidats = list(candidats_uniques.values())
        if not liste_candidats:
            async def generate_empty():
                yield "Je ne trouve pas l'information dans les documents de formation."
            return StreamingResponse(generate_empty(), media_type="text/plain")

        # --- PHASE 4 : RE-RANKING OPTIMISÉ ---
        paires = [[query, candidat["description"]] for candidat in liste_candidats]
        scores = reranker.predict(paires)
        candidats_scores = sorted(zip(scores, liste_candidats), key=lambda x: x[0], reverse=True)
                
        # --- GARDE-FOU HORS-SUJET ---
        meilleur_score = candidats_scores[0][0]
        if meilleur_score < -4.0:
            async def generate_empty():
                yield "[SOURCES]:[]\n"  # On renvoie une liste de sources vide
                yield "Je ne trouve pas l'information dans les documents de formation."
            return StreamingResponse(generate_empty(), media_type="text/plain")
            
        # Passage à un Top 3 pour accélérer le temps de génération (TTFT) du LLM
        top_3_candidats = candidats_scores[:3] 
        
        # --- PHASE 5 : RÉCUPÉRATION DES PARENTS PAR LOTS (Optimisation des requêtes DB) ---
        # 1. On isole tous les IDs parents uniques dont on a besoin
        parent_ids = list(set([candidat["parent_id"] for _, candidat in top_3_candidats]))
        
        # 2. On fait UN UNIQUE appel à Qdrant (Batching)
        points_parents = client.retrieve(collection_name=collection_name, ids=parent_ids)
        dict_parents = {p.id: p.payload.get("global_context", "") for p in points_parents}
        
        contexte_formate = ""
        sources = []
        
        # 3. Construction propre du contexte enrichi
        for i, (score, candidat) in enumerate(top_3_candidats):
            parent_id = candidat["parent_id"]
            contexte_parent = dict_parents.get(parent_id, "Inconnu")
            
            contexte_formate += (
                f"--- Extrait [{i+1}] ---\n"
                f"Fichier Source : {candidat['file_name']}\n"
                f"Contexte Général : {contexte_parent}\n"
                f"Étape/Diapositive : {candidat['step_id']}\n"
                f"Détail de l'étape : {candidat['description']}\n\n"
            )
                            
            sources.append({
                "file_name": candidat["file_name"],
                "step_id": candidat["step_id"],
                "image_path": candidat["image_path"],
                "score_rerank": float(score)
            })

        # --- PHASE 6 : PROMPT SYSTÈME --- [3]
        prompt_systeme = (
            "Rôle : Vous êtes un automate de recherche d'informations techniques.\n"
            "Consigne : Répondez à la question posée uniquement avec les faits du contexte.\n\n"
            "Règles strictes :\n"
            "- Répondez de manière extrêmement directe et concise (pas de formule de politesse, pas d'introduction, pas de conclusion).\n"
            "- Basez-vous uniquement sur les faits écrits dans le CONTEXTE. N'inventez rien et ne supposez rien.\n"
            "- Ne tentez jamais de deviner, d'extrapoler ou d'inventer la signification d'un acronyme (comme PSE, BMR, etc.) si sa définition exacte n'est pas écrite mot pour mot dans le CONTEXTE. Si la définition n'est pas écrite, utilisez uniquement l'acronyme brut."
            "- En cas d'ambiguïté sur l'unité d'un traitement, privilégiez toujours l'unité du principe actif (ex: Unité internationale) plutôt que celle du volume liquide (ex: millilitre).\n"
            "- Si le CONTEXTE ne permet pas de répondre à la question, écrivez uniquement : 'Je ne trouve pas l'information dans les documents de formation.'\n"
            "- Vous devez obligatoirement citer la source de chaque fait important à la fin de la phrase correspondante sous la forme d'un simple numéro d'extrait entre crochets, par exemple [1], [2] ou [3], en vous basant sur le numéro de l'Extrait fourni dans le contexte."
        )
        
        prompt_utilisateur = f"CONTEXTE :\n=========================================\n{contexte_formate}=========================================\n\nQUESTION : {query}"
        
        # --- AMÉLIORATION : INJECTION DYNAMIQUE DE L'HISTORIQUE ---
        messages = [
            {"role": "system", "content": prompt_systeme}
        ]
        
        # On limite l'historique aux 6 derniers messages pour éviter de saturer le contexte du LLM local et ralentir la génération
        for msg in requete.history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})
            
        # On ajoute le message utilisateur courant avec le contexte tout à la fin
        messages.append({"role": "user", "content": prompt_utilisateur})

        # --- PHASE 7 : GÉNÉRATEUR ASYNC DE FLUX SANS TAMPON --- [2, 3]
        async def generate_response():
            # Émettre d'abord les métadonnées de source
            yield f"[SOURCES]:{json.dumps(sources)}\n"
            
            # Utilisation de l'API Asynchrone native d'Ollama
            # Code optimisé :
            async_client = ollama.AsyncClient()
            response_stream = await async_client.chat(
                model=model_id, 
                messages=messages, 
                options={
                    "temperature": 0.0,
                    "num_thread": 10  
                }, 
                stream=True
            )
            async for chunk in response_stream:
                yield chunk['message']['content']

        # Renvoyer le flux avec des en-têtes HTTP de désactivation du cache/tampon [3]
        return StreamingResponse(
            generate_response(), 
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    client.close()
    print("[+] Base Qdrant fermée proprement.")