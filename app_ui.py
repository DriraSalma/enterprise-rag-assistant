import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

# Configuration de la page Web Streamlit
st.set_page_config(
    page_title="ClinicSys Chatbot Assistant",
    page_icon="💬",
    layout="wide"
)

# 1. Chargement des modèles d'IA en cache (pour éviter de les recharger à chaque rafraîchissement d'UI)
@st.cache_resource
def initialiser_moteurs():
    # Connexion au serveur local Qdrant (port 6333) [1.2.3]
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "hth_rag_collection"
    
    # Chargement des modèles sémantiques et de réordonnancement
    model = SentenceTransformer('BAAI/bge-m3')
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    model_id = "mistral" # Modèle de génération Ollama
    
    return client, collection_name, model, reranker, model_id

client, collection_name, model, reranker, model_id = initialiser_moteurs()

# Titre de l'interface graphique
st.title("💬 ClinicSys - Assistant Intelligent de Formation")
st.write("Posez vos questions sur l'utilisation de l'interface des logiciels cliniques.")

# Initialiser l'historique de discussion dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages de l'historique de discussion
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"], caption="Support visuel associé", width=500)

# Zone de saisie de l'utilisateur
if query := st.chat_input("Posez votre question technique ici (ex: Que signifie la mention BMR ?)..."):
    
    # Afficher la question de l'utilisateur dans l'interface
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Lancement de la recherche et de la génération
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # --- PHASE 1 : RECHERCHE DENSE (SEMANTIQUE) ---
        query_vector = model.encode(query, convert_to_numpy=True).tolist()
        resultats_dense = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="child"))]),
            limit=15
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
            limit=15
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
            reponse_erreur = "Je ne trouve pas l'information dans les documents de formation."
            message_placeholder.markdown(reponse_erreur)
            st.session_state.messages.append({"role": "assistant", "content": reponse_erreur})
        else:
            # --- PHASE 4 : RE-RANKING (Cross-Encoder) ---
            paires = [[query, candidat["description"]] for candidat in liste_candidats]
            scores = reranker.predict(paires)
            candidats_scores = sorted(zip(scores, liste_candidats), key=lambda x: x[0], reverse=True)
            
            # Alignement parfait avec app.py : on prend le Top 12
            top_12_candidats = candidats_scores[:12]
            
            # --- PHASE 5 : RÉCUPÉRATION DU CONTEXTE ---
            contexte_formate = ""
            image_path_to_display = None
            
            for i, (score, candidat) in enumerate(top_12_candidats):
                parent_id = candidat["parent_id"]
                points_parent = client.retrieve(collection_name=collection_name, ids=[parent_id])
                contexte_parent = points_parent[0].payload.get("global_context", "") if points_parent else "Inconnu"
                
                contexte_formate += (
                    f"--- Extrait {i+1} ---\n"
                    f"Fichier Source : {candidat['file_name']}\n"
                    f"Étape/Diapositive : {candidat['step_id']}\n"
                    f"Détail de l'étape : {candidat['description']}\n\n"
                )
                
                # On récupère le chemin de l'image du meilleur candidat trouvé (Rang 1)
                if i == 0:
                    image_path_to_display = candidat.get("image_path")

            # --- PHASE 6 : GÉNÉRATION AVEC OLLAMA (STREAMING) ---
            prompt_systeme = (
                "Rôle : Vous êtes un assistant d'apprentissage technique pour les logiciels de la société ClinicSys.\n"
                "Votre unique tâche est de répondre à la QUESTION de l'utilisateur à partir des faits présents dans le CONTEXTE fourni.\n\n"
                "Consignes absolues :\n"
                "- Répondez de manière extrêmement directe et concise (pas de formule de politesse, pas d'introduction, pas de conclusion).\n"
                "- Basez-vous uniquement sur les faits écrits dans le CONTEXTE. N'inventez rien et ne supposez rien.\n"
                "- En cas d'ambiguïté sur l'unité d'un traitement, privilégiez toujours l'unité du principe actif (ex: Unité internationale) plutôt que celle du volume liquide (ex: millilitre).\n"
                "- Si le CONTEXTE ne permet pas de répondre à la question, écrivez uniquement : 'Je ne trouve pas l'information dans les documents de formation.'\n"
                "- Vous devez obligatoirement ajouter à la fin de chaque phrase de votre réponse sa source correspondante au format exact : [Nom réel du fichier, Étape/Diapositive] (Exemple de format : [project_1-1.json, 52])."
            )
            
            prompt_utilisateur = f"CONTEXTE :\n=========================================\n{contexte_formate}=========================================\n\nQUESTION : {query}"
            
            messages = [
                {"role": "system", "content": prompt_systeme},
                {"role": "user", "content": prompt_utilisateur}
            ]
            
            # Lancer la génération en flux (Streaming) [3]
            response_stream = ollama.chat(model=model_id, messages=messages, options={"temperature": 0.0}, stream=True)
            
            reponse_complete = ""
            for chunk in response_stream:
                reponse_complete += chunk['message']['content']
                # Afficher progressivement le texte s'écrire lettre par lettre dans l'interface web [3]
                message_placeholder.markdown(reponse_complete + "▌")
                
            # Afficher le texte final propre
            message_placeholder.markdown(reponse_complete)
            
            # --- PHASE 7 : AFFICHAGE DE LA MEILLEURE CAPTURE D'ÉCRAN ---
            # Si le dossier local contient l'image, on l'affiche directement sous le texte
            if image_path_to_display and os.path.exists(image_path_to_display):
                st.image(image_path_to_display, caption=f"Support visuel : {image_path_to_display}", width=500)
            
            # Sauvegarder la réponse complète et l'image dans l'historique de discussion
            st.session_state.messages.append({
                "role": "assistant", 
                "content": reponse_complete,
                "image": image_path_to_display if (image_path_to_display and os.path.exists(image_path_to_display)) else None
            })