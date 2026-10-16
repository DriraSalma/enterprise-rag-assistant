import os
import json
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

def ingest_hth_rag(dossier_cleaned, dossier_db):
    chemin_cleaned = Path(dossier_cleaned)
    chemin_db = Path(dossier_db)
    
    # 1. Initialiser Qdrant en mode local (sauvegardé sur le disque)
    #print("[+] Initialisation de la base vectorielle locale Qdrant...")
    #client = QdrantClient(path=str(chemin_db))

    print("[+] Connexion au serveur local Qdrant...")
    client = QdrantClient(url="http://localhost:6333") # Connexion au serveur
    
    # 2. Charger le modèle d'embeddings BAAI/bge-m3 (modèle de pointe)
    print("[+] Chargement du modèle d'embeddings BAAI/bge-m3...")
    # Ce modèle de 1024 dimensions est extrêmement puissant pour le français
    model = SentenceTransformer('BAAI/bge-m3')
    dimension_vecteurs = 1024
    
    # Créer la collection Qdrant (elle sera écrasée à chaque ré-ingestion pour repartir de zéro)
    collection_name = "hth_rag_collection"
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension_vecteurs, distance=Distance.COSINE),
    )
    
    fichiers = [f for f in chemin_cleaned.iterdir() if f.suffix in ['.json', '.txt']]
    if not fichiers:
        print(f"[-] Aucun fichier nettoyé trouvé dans '{dossier_cleaned}'.")
        return

    print(f"[+] Traitement et indexation de {len(fichiers)} fichiers...")
    
    points_to_insert = []
    nb_parents = 0
    nb_enfants = 0
    
    for chemin_fichier in fichiers:
        try:
            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                etapes = json.load(f)
            
            if not isinstance(etapes, list) or len(etapes) == 0:
                continue
                
            # --- PHASE A : COMPILATION DU TEXTE PARENT ---
            nom_fichier = chemin_fichier.name
            parent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, nom_fichier))
            
            # Concaténation propre de toutes les descriptions pour créer le contexte global
            texte_global_parent = " ".join([
                e.get("description", "").strip().replace("\n", " ") 
                for e in etapes if e.get("description")
            ])
            
            # Générer l'embedding du parent
            embedding_parent = model.encode(texte_global_parent, convert_to_numpy=True).tolist()
            
            # Ajouter le Parent dans les points à insérer
            points_to_insert.append(
                PointStruct(
                    id=parent_id,
                    vector=embedding_parent,
                    payload={
                        "type": "parent",
                        "file_name": nom_fichier,
                        "global_context": texte_global_parent
                    }
                )
            )
            nb_parents += 1
            
            # --- PHASE B : EXTRACTION ET ENRICHISSEMENT DES ENFANTS ---
            for etape in etapes:
                description_enfant = etape.get("description", "").strip()
                if not description_enfant:
                    continue
                    
                enfant_id = str(uuid.uuid4())
                
                # Enrichissement : on fusionne le titre/contexte global avec l'étape pour conserver le sens
                texte_enrichi_enfant = (
                    f"Fichier : {nom_fichier} | "
                    f"Contexte global : {texte_global_parent[:250]}... | "
                    f"Étape spécifique : {description_enfant}"
                )
                
                # Générer l'embedding de l'enfant enrichi
                embedding_enfant = model.encode(texte_enrichi_enfant, convert_to_numpy=True).tolist()
                
                # Ajouter l'Enfant dans les points à insérer avec ses métadonnées
                points_to_insert.append(
                    PointStruct(
                        id=enfant_id,
                        vector=embedding_enfant,
                        payload={
                            "type": "child",
                            "parent_id": parent_id,
                            "file_name": nom_fichier,
                            "step_id": etape.get("id"),
                            "image_path": etape.get("path"),
                            "description": description_enfant
                        }
                    )
                )
                nb_enfants += 1
                
        except Exception as e:
            print(f"[!] Erreur de traitement sur {chemin_fichier.name} : {e}")

    # 3. Insertion par lots (batching) dans Qdrant
    if points_to_insert:
        print(f"[+] Insertion de {len(points_to_insert)} points dans Qdrant (Parents: {nb_parents}, Enfants: {nb_enfants})...")
        client.upsert(
            collection_name=collection_name,
            points=points_to_insert
        )
        print("[+] Indexation HTH-RAG terminée avec succès !")
    else:
        print("[-] Aucun point valide à insérer.")

if __name__ == "__main__":
    DOSSIER_CLEANED = "data/cleaned"
    DOSSIER_DB = "data/qdrant_db"
    
    ingest_hth_rag(DOSSIER_CLEANED, DOSSIER_DB)