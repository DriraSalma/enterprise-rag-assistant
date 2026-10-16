import time
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

def run_benchmark(query):
    print("[+] Connexion à la base locale Qdrant...")
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "hth_rag_collection"

    # Charger les modèles d'IA en local
    print("[+] Chargement du modèle sémantique BGE-M3...")
    model = SentenceTransformer('BAAI/bge-m3')
    
    print("[+] Chargement du modèle de Re-ranking (Cross-Encoder)...")
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    model_id = "mistral"

    print(f"\n[+] DÉBUT DU BENCHMARK POUR : '{query}'")

    # ================= PHASE 1 : MESURE DU RETRIEVAL HYBRIDE (Qdrant) =================
    start_retrieval = time.perf_counter()
    
    # 1. Recherche Dense (Sémantique)
    query_vector = model.encode(query, convert_to_numpy=True).tolist()
    resultats_dense = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="child"))]),
        limit=15
    )
    
    # 2. Recherche Sparse (Lexicale)
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
    
    # 3. Fusion et Dédoublonnage
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
    
    end_retrieval = time.perf_counter()
    time_retrieval_ms = (end_retrieval - start_retrieval) * 1000 # Temps en millisecondes

    # ================= PHASE 2 : MESURE DU RE-RANKING (Cross-Encoder) =================
    if not liste_candidats:
        print("[-] Aucun candidat trouvé pour exécuter le benchmark.")
        return
        
    start_rerank = time.perf_counter()
    
    paires = [[query, candidat["description"]] for candidat in liste_candidats]
    scores = reranker.predict(paires)
    candidats_scores = sorted(zip(scores, liste_candidats), key=lambda x: x[0], reverse=True)
    top_12_candidats = candidats_scores[:5]
    
    end_rerank = time.perf_counter()
    time_rerank_ms = (end_rerank - start_rerank) * 1000 # Temps en millisecondes

    # ================= PHASE 3 : MESURE DE LA GÉNÉRATION DE TEXTE (LLM) =================
    # Assemblage du contexte
    contexte_formate = ""
    for i, (score, candidat) in enumerate(top_12_candidats):
        parent_id = candidat["parent_id"]
        points_parent = client.retrieve(collection_name=collection_name, ids=[parent_id])
        contexte_parent = points_parent[0].payload.get("global_context", "") if points_parent else "Inconnu"
        contexte_formate += f"--- Extrait {i+1} ---\nFichier : {candidat['file_name']} | Étape : {candidat['step_id']}\n{candidat['description']}\n\n"

    prompt_systeme = (
        "Vous êtes un assistant technique. Répondez à la QUESTION de manière courte et directe en utilisant uniquement le CONTEXTE."
    )
    prompt_utilisateur = f"CONTEXTE :\n{contexte_formate}\n\nQUESTION : {query}"
    messages = [{"role": "system", "content": prompt_systeme}, {"role": "user", "content": prompt_utilisateur}]
    
    # Mesurer la génération complète du LLM (sans streaming pour mesurer le temps total) [3]
    start_generation = time.perf_counter()
    response = ollama.chat(model=model_id, messages=messages, options={"temperature": 0.0})
    end_generation = time.perf_counter()
    
    time_generation_s = end_generation - start_generation # Temps en secondes
    
    reponse_texte = response['message']['content']
    
    # Estimation de la vitesse de génération (en tokens par seconde)
    # Règle empirique : 1 mot en français vaut environ 1.3 tokens
    nb_mots = len(reponse_texte.split())
    nb_tokens_estimes = int(nb_mots * 1.3)
    tokens_per_second = nb_tokens_estimes / time_generation_s if time_generation_s > 0 else 0

    # ================= AFFICHAGE DES RÉSULTATS DU BENCHMARK =================
    print("\n======================= RÉSULTATS DU BENCHMARK =======================")
    print(f"1. Temps de Recherche Hybride (Qdrant)  : {time_retrieval_ms:.2f} ms")
    print(f"2. Temps de Re-ranking (Cross-Encoder)  : {time_rerank_ms:.2f} ms")
    print(f"3. Temps d'Inférence LLM (Ollama CPU)   : {time_generation_s:.2f} secondes")
    print(f"4. Longueur de la réponse générée      : {nb_mots} mots (environ {nb_tokens_estimes} tokens)")
    print(f"5. Débit de Génération (Vitesse)        : {tokens_per_second:.2f} tokens/seconde")
    print("======================================================================")
    print(f"\nRéponse générée :\n{reponse_texte}\n")

if __name__ == "__main__":
    # Question de test
    query_test = "Quelle est la seule différence de fonctionnement entre la section 'Demande radio' et la section 'Demande labo'?"
    run_benchmark(query_test)