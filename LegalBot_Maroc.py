# %%
!pip
install - q
neo4j
# %%
from neo4j import GraphDatabase
import json

# Configuration d'après tes infos
URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "12345678"
DATABASE = "projet"
JSON_PATH = r"C:\ProjetAI\legal_chunks.json"  #


def remplir_base_graphe():
    # 1. Charger les données JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 2. Se connecter au driver
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session(database=DATABASE) as session:
        print(f"🚀 Injection de {len(chunks)} articles dans la base '{DATABASE}'...")

        # Requête Cypher pour créer les nœuds Source et Article
        query = """
        UNWIND $data AS item
        MERGE (s:Source {name: item.source})
        CREATE (a:Article {
            article_num: item.article,
            content: item.text
        })
        CREATE (a)-[:APPARTIENT_A]->(s)
        """
        session.run(query, data=chunks)

    driver.close()
    print("✅ Migration terminée avec succès !")


# Exécuter la fonction
remplir_base_graphe()
# %%
import re
from neo4j import GraphDatabase

# Configuration (identique à ton étape précédente)
URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "12345678"
DATABASE = "projet"


def creer_citations_automatiques():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session(database=DATABASE) as session:
        print("🔍 Analyse du texte pour trouver des citations...")

        # 1. On récupère tous les articles
        articles = session.run("MATCH (a:Article) RETURN a.article_num as num, id(a) as id, a.content as text")

        for record in articles:
            texte = record["text"]
            id_source = record["id"]

            # 2. On cherche des motifs comme "Article 45" ou "art. 45"
            # Ce pattern est adapté au format des textes juridiques marocains
            citations = re.findall(r"(?:Article|art\.)\s+(\d+)", texte, re.IGNORECASE)

            for num_cite in citations:
                # 3. On crée un lien vers l'article mentionné s'il existe dans la même source
                session.run("""
                    MATCH (a1) WHERE id(a1) = $id_source
                    MATCH (a2:Article {article_num: $num_cite})
                    WHERE a1 <> a2
                    MERGE (a1)-[:CITE]->(a2)
                """, id_source=id_source, num_cite=num_cite)

    driver.close()
    print("✅ Liens de citations créés dans le graphe !")


# Lancer la détection
creer_citations_automatiques()
# %%
# 0) Installer les dépendances (à relancer après chaque changement de runtime)
!pip
install - q
faiss - cpu
sentence - transformers
transformers
accelerate

import os, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

# %%
import os

BASE_DIR = r"C:\ProjetAI"
print("BASE_DIR existe ?", os.path.exists(BASE_DIR))
print("Contenu de BASE_DIR :", os.listdir(BASE_DIR))

# %%
# 2) Indiquer le dossier où se trouvent les fichiers sur Drive
# Exemple : "/content/drive/MyDrive/legal_bot_db"
DRIVE_DB_PATH = "C:\ProjetAI"  # <-- change ce chemin si besoin

index_path = f"{DRIVE_DB_PATH}/legal_index.faiss"
chunks_path = f"{DRIVE_DB_PATH}/legal_chunks.json"

print("Index path :", index_path)
print("Chunks path:", chunks_path)

# Vérification du dossier
!ls - lh
"$DRIVE_DB_PATH"

# %%
# 3) Charger l'embedder (le même que pour l'indexation)
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

# %%
# 4) Charger l'index FAISS + les chunks depuis Drive
index = faiss.read_index(index_path)

with open(chunks_path, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print("✅ Index chargé | nb chunks:", len(all_chunks))
print("✅ Sources dispo:", sorted(set(c.get("source") for c in all_chunks if c.get("source"))))


# %% md
## 🔎 Retrieval (recherche sémantique)
# %%
def retrieve(question, k=5, source=None, min_score=0.25):
    """Retourne top-k chunks pertinents.
    - source: filtrer un code particulier (ex: 'droit_penal')
    - min_score: seuil pour éviter hallucinations
    """
    q_emb = embedder.encode([f"query: {question}"], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    k_search = k * 5 if source else k
    scores, idxs = index.search(q_emb, k_search)

    results = []
    for i, score in zip(idxs[0], scores[0]):
        if score < min_score:
            continue
        chunk = all_chunks[i]
        if (source is None) or (chunk.get("source") == source):
            results.append((chunk, float(score)))
        if len(results) >= k:
            break
    return results


# %% md
## 🤖 Charger un LLM Hugging Face
Choisis
un
modèle
instruct.Si
tu
es
sur
CPU, prends
un
modèle
léger.
Exemples
CPU - friendly: `Qwen / Qwen2
.5 - 1.5
B - Instruct
`, `microsoft / phi - 2`.
Ici
on
met
Mistral
7
B
Instruct(GPU
recommandé).
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=None,  # CPU
    torch_dtype=torch.float32  # CPU
)

gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
gen.tokenizer.pad_token_id = gen.tokenizer.eos_token_id

print(gen("Question: Quelle est la peine du vol ?\nRéponse:", max_new_tokens=150)[0]["generated_text"])


# %% md
## 🧠 RAG Answer
# %%
def get_graph_context(retrieved_chunks):
    """Pour chaque chunk trouvé par FAISS, on va chercher ses citations dans Neo4j."""
    driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "12345678"))
    extra_chunks = []

    # On mémorise les articles déjà présents pour éviter les doublons
    seen_articles = {str(c.get('article')) for c, score in retrieved_chunks}

    with driver.session(database="projet") as session:
        for chunk, score in retrieved_chunks:
            art_num = str(chunk.get('article'))

            # Requête Cypher : trouve les articles cités par l'article actuel
            result = session.run("""
                MATCH (a:Article {article_num: $num})-[:CITE]->(cite:Article)
                RETURN cite.article_num as num, cite.content as text
            """, num=art_num)

            for record in result:
                if record["num"] not in seen_articles:
                    # On crée un faux chunk pour l'article cité
                    extra_chunks.append({
                        'source': 'Référence liée (Graphe)',
                        'article': record["num"],
                        'text': record["text"]
                    })
                    seen_articles.add(record["num"])

    driver.close()
    return extra_chunks


# %%
def rag_answer_v2(question, k=5, source=None, min_score=0.25, max_ctx_chars=6000):
    # 1. Recherche initiale avec FAISS (votre fonction actuelle)
    retrieved = retrieve(question, k=k, source=source, min_score=min_score)

    if not retrieved:
        return "Je ne sais pas. Aucun article pertinent trouvé.", []

    # 2. EXTENSION GRAPHE : On va chercher les citations dans Neo4j
    extra_context = get_graph_context(retrieved)

    # 3. Construction du contexte enrichi
    parts = []
    total = 0

    # On ajoute d'abord les articles principaux (FAISS)
    for c, score in retrieved:
        chunk = f"[{c['source']} | Article {c.get('article')} | score={score:.2f}] {c['text']}"
        if total + len(chunk) < max_ctx_chars:
            parts.append(chunk)
            total += len(chunk)

    # On ajoute ensuite les articles liés par le graphe (Neo4j)
    for c in extra_context:
        chunk = f"[LIEN | Article {c.get('article')}] {c['text']}"
        if total + len(chunk) < max_ctx_chars:
            parts.append(chunk)
            total += len(chunk)

    context = "\n\n".join(parts)

    # 4. Prompt et Génération avec TinyLlama
    prompt = f"""
Tu es un assistant juridique marocain.
Réponds à la QUESTION en utilisant le CONTEXTE (articles principaux et liés).

CONTEXTE:
{context}

QUESTION:
{question}

RÉPONSE:
"""

    out = gen(prompt, max_new_tokens=450, do_sample=False)[0]["generated_text"]
    answer = out.split("RÉPONSE")[-1].strip()

    return answer, retrieved + extra_context


# %% md
## ✅ Tests rapides
# %%
# 1. Définition de la question de test
question_test = "Quelle est la procédure à suivre pour un licenciement pour faute grave ?"

# 2. Exécution du GraphRAG
print("🔍 Recherche en cours (FAISS + Neo4j)...")
reponse, sources = rag_answer_v2(question_test, k=4)

# 3. Affichage de la réponse
print("\n" + "=" * 50)
print("🤖 RÉPONSE DU LEGALBOT :")
print("=" * 50)
print(reponse)

print("\n" + "=" * 50)
print("📚 SOURCES UTILISÉES :")
print("=" * 50)
for i, s in enumerate(sources):
    # On distingue les sources FAISS des sources Graph
    type_source = "Graphe (Lien)" if isinstance(s, dict) and s.get(
        'source') == 'Référence liée (Graphe)' else "FAISS (Similarité)"

    # Extraction propre des infos selon le type
    if isinstance(s, tuple):  # Format FAISS: (chunk_dict, score)
        chunk, score = s
        print(f"{i + 1}. [{type_source}] Article {chunk.get('article')} (Score: {score:.2f})")
    else:  # Format Graphe
        print(f"{i + 1}. [{type_source}] Article {s.get('article')}")
# %%
!pip
install - q
gradio
# %%
import gradio as gr


def bot_interface(question, top_k, temperature):
    """Fonction de pont entre l'interface et votre code GraphRAG."""

    # 1. Appel de votre fonction RAG avec les paramètres de l'interface
    # On modifie temporairement rag_answer_v2 pour accepter la température
    answer, sources = rag_answer_v2_ui(question, k=int(top_k), temp=float(temperature))

    # 2. Formatage des sources pour l'affichage
    sources_text = "\n\n".join([
        f"📌 {s[0]['source']} (Art. {s[0].get('article')})" if isinstance(s, tuple)
        else f"🔗 {s.get('source')} (Art. {s.get('article')})"
        for s in sources
    ])

    return answer, sources_text


def rag_answer_v2_ui(question, k=5, temp=0.1):
    """Version adaptée pour l'interface avec gestion de la température."""
    retrieved = retrieve(question, k=k)  # Utilise votre fonction FAISS

    if not retrieved:
        return "Aucun article trouvé.", []

    extra_context = get_graph_context(retrieved)  # Utilise Neo4j

    # Construction du contexte (identique à votre version précédente)
    parts = []
    for c, score in retrieved:
        parts.append(f"[{c['source']} | Art. {c.get('article')}] {c['text']}")
    for c in extra_context:
        parts.append(f"[LIEN | Art. {c.get('article')}] {c['text']}")

    context = "\n\n".join(parts)
    prompt = f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}\n\nRÉPONSE:"

    # Génération avec prise en compte de la température
    # do_sample=True est nécessaire pour utiliser la température
    out = gen(prompt, max_new_tokens=450, do_sample=True, temperature=temp)[0]["generated_text"]
    answer = out.split("RÉPONSE")[-1].strip()

    return answer, retrieved + extra_context


# %%
# Configuration de l'interface visuelle
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇲🇦 LegalBot - Assistant Juridique GraphRAG")
    gr.Markdown("Posez vos questions sur le droit marocain (Code du Travail, Commerce, etc.)")

    with gr.Row():
        with gr.Column(scale=1):
            # Paramètres de contrôle
            input_text = gr.Textbox(label="Votre question", placeholder="Ex: Quelle est la durée du préavis ?")
            slider_k = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Top-K (Articles FAISS)")
            slider_temp = gr.Slider(minimum=0.1, maximum=1.0, value=0.1, step=0.1, label="Température (Créativité)")
            btn = gr.Button("Rechercher 🔍", variant="primary")

        with gr.Column(scale=2):
            # Affichage des résultats
            output_answer = gr.Textbox(label="Réponse du LegalBot", lines=10)
            output_sources = gr.Textbox(label="Sources juridiques consultées (FAISS + Neo4j)", lines=5)

    # Action du bouton
    btn.click(fn=bot_interface, inputs=[input_text, slider_k, slider_temp], outputs=[output_answer, output_sources])

# Lancement local (accessible sur http://127.0.0.1:7860)
demo.launch(share=False)