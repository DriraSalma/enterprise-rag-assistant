import os
import sys # 1. Importation de la bibliothèque système pour l'affichage progressif
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

def initialiser_moteurs():
    #print("[+] Connexion à la base vectorielle locale Qdrant...")
    #client = QdrantClient(path="data/qdrant_db")
    print("[+] Connexion au serveur Qdrant...")
    client = QdrantClient(url="http://localhost:6333") # Connexion au serveur Qdrant 

    collection_name = "hth_rag_collection"
    
    print("[+] Chargement du modèle sémantique BGE-M3...")
    model = SentenceTransformer('BAAI/bge-m3')
    
    print("[+] Chargement du modèle de Re-ranking (Cross-Encoder)...")
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    
    # Remplacement du modèle Qwen par Mistral 7B (technologie française)
    model_id = "mistral"
    
    #model_id = "mistral:7b-instruct-q2_K" # Version ultra-légère de Mistral 7B (3,1 Go)
    print(f"[+] Connexion au Générateur Local via Ollama ({model_id})...")
    
    return client, collection_name, model, reranker, model_id

def executer_hth_rag(query, client, collection_name, model, reranker, model_id):
    # --- PHASE 1 : RECHERCHE DENSE (SEMANTIQUE) ---
    query_vector = model.encode(query, convert_to_numpy=True).tolist()
    resultats_dense = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="child"))]),
        limit=15
    )
    
    # --- PHASE 2 : RECHERCHE SPARSE / LEXICALE (TEXT MATCH) ---
    resultats_sparse, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="type", match=MatchValue(value="child")),
                FieldCondition(key="description", match=MatchText(text=query))
            ]
        ),
        limit=15
    )
    
    # --- PHASE 3 : FUSION ET DÉDOUBLONNAGE CORRECTS ---
    # Nous utilisons la description textuelle (string) comme clé pour éviter le TypeError: unhashable type: 'dict'
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
        return "Je ne trouve pas l'information dans les documents de formation."

    # --- PHASE 4 : RE-RANKING (CROSS-ENCODER) ---
    paires = [[query, candidat["description"]] for candidat in liste_candidats]
    scores = reranker.predict(paires)
    candidats_scores = sorted(zip(scores, liste_candidats), key=lambda x: x[0], reverse=True)
    
    top_3_candidats = candidats_scores[:5]
    
    # --- PHASE 5 : RÉCUPÉRATION DU CONTEXTE ---
    contexte_formate = ""
    for i, (score, candidat) in enumerate(top_3_candidats):
        parent_id = candidat["parent_id"]
        points_parent = client.retrieve(collection_name=collection_name, ids=[parent_id])
        contexte_parent = points_parent[0].payload.get("global_context", "") if points_parent else "Inconnu"
        
        contexte_formate += (
            f"--- Extrait {i+1} ---\n"
            f"Fichier Source : {candidat['file_name']}\n"
            f"Étape/Diapositive : {candidat['step_id']}\n"
            f"Détail de l'étape : {candidat['description']}\n\n"
        )

    # ==================== [DEBUG] AFFICHAGE DU CONTEXTE ====================
    print("\n=================== [DEBUG] CONTEXTE RETROUVÉ ===================")
    if not contexte_formate.strip():
        print("[!] ALERTE : Le contexte est VIDE. Qdrant n'a rien trouvé.")
    else:
        print(contexte_formate)
    print("=================================================================\n")
    # =======================================================================

    # 6. Prompt Système Strict d'Ancrage Contextuel (Optimisé pour les modèles de 7B) [3]
   # 6. Prompt Système Strict (Corrigé pour forcer le remplacement du nom de fichier) [3]
    prompt_systeme = (
        "Rôle : Vous êtes un assistant d'apprentissage technique pour les logiciels de la société ClinicSys.\n"
        "Votre unique tâche est de répondre à la QUESTION de l'utilisateur à partir des faits présents dans le CONTEXTE fourni.\n\n"
        "Consignes absolues :\n"
        "- Répondez de manière extrêmement directe et concise (pas de formule de politesse, pas d'introduction, pas de conclusion).\n"
        "- Basez-vous uniquement sur les faits écrits dans le CONTEXTE. N'inventez rien et ne supposez rien.\n"
        "- En cas d'ambiguïté sur l'unité d'un traitement, privilégiez toujours l'unité du principe actif (ex: Unité internationale) plutôt que celle du volume liquide (ex: millilitre).\n" # <-- AJOUT DE CETTE RÈGLE DE PRIORITÉ [3]
        "- Si le CONTEXTE ne permet pas de répondre à la question, écrivez uniquement : 'Je ne trouve pas l'information dans les documents de formation.'\n"
        "- Vous devez obligatoirement ajouter à la fin de chaque phrase de votre réponse sa source correspondante au format exact : [Nom réel du fichier, Étape/Diapositive] (Exemple de format : [project_1-1.json, 52])."
    )
    
    prompt_utilisateur = f"CONTEXTE :\n=========================================\n{contexte_formate}=========================================\n\nQUESTION : {query}"
    
    messages = [
        {"role": "system", "content": prompt_systeme},
        {"role": "user", "content": prompt_utilisateur}
    ]
    
    

    # 7. Génération via l'API locale en mode STREAMING [3]
    print("[+] Génération de la réponse finale par Ollama...")
    print("\n================ RÉPONSE DU CHATBOT RAG ================")
    
    response_stream = ollama.chat(
        model=model_id, 
        messages=messages,
        options={"temperature": 0.0},
        stream=True # Activer le mode streaming [3]
    )
    
    # Boucle pour afficher chaque mot/lettre dès qu'il est calculé
    for chunk in response_stream:
        content = chunk['message']['content']
        sys.stdout.write(content)
        sys.stdout.flush() # Force le rafraîchissement d'écriture de la console en temps réel

if __name__ == "__main__":
    try:
        client, collection_name, model, reranker, model_id = initialiser_moteurs()
        
        while True:
            query = input("\nPosez votre question (ou tapez 'exit' pour quitter) : ")
            if query.lower() == 'exit':
                break
            
            reponse = executer_hth_rag(query, client, collection_name, model, reranker, model_id)
            #print("\n================ RÉPONSE DU CHATBOT RAG ================")
            #print(reponse)
            # print("========================================================")
            
    finally:
        # Fermeture propre de la base pour éviter le blocage de port au redémarrage
        if 'client' in locals():
            client.close()
            print("[+] Base Qdrant fermée proprement.")