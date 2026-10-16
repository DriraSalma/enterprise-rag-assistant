import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from sentence_transformers import SentenceTransformer, CrossEncoder

def initialiser_moteurs():
    # 1. Connexion à Qdrant
    client = QdrantClient(path="data/qdrant_db")
    collection_name = "hth_rag_collection"
    
    # 2. Charger le modèle d'embeddings BGE-M3 (pour la recherche dense)
    print("[+] Chargement du modèle sémantique BGE-M3...")
    model = SentenceTransformer('BAAI/bge-m3')
    
    # 3. Charger le modèle de Re-ranking local (Cross-Encoder)
    print("[+] Chargement du modèle de Re-ranking (Cross-Encoder)...")
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    
    return client, collection_name, model, reranker

def recherche_hybride_hth(query, client, collection_name, model, reranker):
    print(f"\n[+] Recherche en cours pour : '{query}'")
    
    # --- PHASE 1 : RECHERCHE DENSE (SEMANTIQUE) ---
    # Nous utilisons la nouvelle méthode de l'API Qdrant : query_points()
    query_vector = model.encode(query, convert_to_numpy=True).tolist()
    resultats_dense = client.query_points(
        collection_name=collection_name,
        query=query_vector, # query_vector remplacé par query
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
    
    # --- PHASE 3 : FUSION ET DÉDOUBLONNAGE (TOP 15) ---
    candidats_uniques = {}
    
    # Ajouter les résultats de la recherche dense (via l'attribut .points de query_points)
    for hit in resultats_dense.points:
        candidats_uniques[hit.id] = hit.payload
        
    # Ajouter les résultats de la recherche lexicale (BM25-like)
    for point in resultats_sparse:
        if point.id not in candidats_uniques:
            candidats_uniques[point.id] = point.payload
            
    liste_candidats = list(candidats_uniques.values())
    if not liste_candidats:
        print("[-] Aucun résultat trouvé.")
        return []

    # --- PHASE 4 : RE-RANKING (CROSS-ENCODER) ---
    print(f"[+] Re-ranking de {len(liste_candidats)} candidats potentiels...")
    
    # Préparer les paires [Question, Description du segment] pour le Cross-Encoder
    paires = [[query, candidat["description"]] for candidat in liste_candidats]
    scores = reranker.predict(paires)
    
    # Associer les scores aux candidats et trier par pertinence (ordre décroissant)
    candidats_scores = sorted(zip(scores, liste_candidats), key=lambda x: x[0], reverse=True)
    
    # Conserver uniquement le Top 3
    top_3_candidats = candidats_scores[:5]
    
    # --- PHASE 5 : ENRICHISSEMENT AVEC LE CONTEXTE PARENT ---
    resultats_finals = []
    print("\n[+] Résultats les plus pertinents (Top 3 avec Re-ranking) :")
    
    for i, (score, candidat) in enumerate(top_3_candidats):
        parent_id = candidat["parent_id"]
        
        # Récupérer le contexte global du Parent associé
        points_parent = client.retrieve(
            collection_name=collection_name,
            ids=[parent_id]
        )
        contexte_parent = points_parent[0].payload.get("global_context", "") if points_parent else "Inconnu"
        
        resultats_finals.append({
            "score_rerank": float(score),
            "description": candidat["description"],
            "step_id": candidat["step_id"],
            "image_path": candidat["image_path"],
            "file_name": candidat["file_name"],
            "global_context": contexte_parent
        })
        
        print(f"\nRang {i+1} [Score Re-ranker: {score:.4f}]")
        print(f"  -> Source : {candidat['file_name']} (Étape : {candidat['step_id']})")
        print(f"  -> Image : {candidat['image_path']}")
        print(f"  -> Extrait trouvé : \"{candidat['description']}\"")
        
    return resultats_finals

if __name__ == "__main__":
    client, collection_name, model, reranker = initialiser_moteurs()
    
    # Exemple de requête à tester
    #requete_test = "Où s'affichent les dates de passage des médecins ?"
    #requete_test = "quand on peut visualiser l’ensemble des tâche?"
    requete_test="À quoi sert le raccourci Compte rendu ?"
    recherche_hybride_hth(requete_test, client, collection_name, model, reranker)