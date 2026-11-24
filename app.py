
import streamlit as st

from langchain_core.messages import AIMessageChunk


# ============================================================
# 1. CONFIGURATION DE UI AVEC STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Datheon AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CHARGEMENT DE NOTRE L'AGENT IA
# ============================================================

@st.cache_resource
def load_agent():

    # Importer l'agent existant de task3.py
    from task3 import agent, MODEL_NAME

    return agent, MODEL_NAME


try:
    agent, model_name = load_agent()
    agent_available = True

except Exception as error:

    agent_available = False
    st.error(f"Impossible de charger l'agent : {error}")


# ============================================================
# 3. INITIALISATION DE LA SESSION DE MON SYSTEME
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "processing" not in st.session_state:

    st.session_state.processing = False


# ============================================================
# 4. BARRE de MENU LATERALE
# ============================================================

with st.sidebar:

    st.title("🤖 Datheon AI")

    st.caption("Multi-Tool Autonomous AI Assistant")

    st.divider()

    st.subheader("Configuration")

    if agent_available:

        st.success("Agent connecté")

        st.write(f"**Modèle :** {model_name}")

    else:

        st.error("Agent indisponible")

    st.divider()

    st.subheader("Outils disponibles")

    st.markdown("""
    🌐 **Web Search**

    Recherche sur Internet avec DuckDuckGo.

    🧮 **Python Calculator**

    Calculs mathématiques sécurisés.

    📁 **Local File Reader**

    Lecture des fichiers locaux.
    """)

    st.divider()

    st.subheader("Historique")

    st.write(
        f"Messages : {len(st.session_state.messages)}"
    )

    if st.button(
        "🗑️ Effacer la conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "Développé avec Python, Streamlit, "
        "LangChain et Google Gemini,"
         "Par Prech Azael Bivihou."
    )


# ============================================================
# 5. EN-TETE PRINCIPAL
# ============================================================

st.title("🤖 Datheon AI Assistant")

st.markdown(
    """
    Bienvenue dans votre assistant intelligent.

    Posez vos questions et laissez l'agent choisir
    automatiquement les outils nécessaires pour vous répondre.
    """
)

st.divider()


# ===============================================================
# 6. AFFICHAGE DE L'HISTORIQUE DE VOS CONVERSATIONS AVEC L'AGENT
# ===============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# 7. FONCTION DE STREAMING
# ============================================================

def stream_agent_response():

    """
    Transmet les messages de l'agent progressivement.

    L'historique complet est envoyé à LangChain
    pour conserver le contexte conversationnel.
    """

    conversation = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages
    ]

    # Streaming LangChain
    for chunk, metadata in agent.stream(
        {
            "messages": conversation
        },
        stream_mode="messages",
        config={
            "recursion_limit": 15
        }
    ):

        # Ici, on ne va récupérer que les fragments textuels
        # des réponses du modèle, pas les appels d'outils.

        if isinstance(chunk, AIMessageChunk):

            content = chunk.content

            if isinstance(content, str):

                if content:
                    yield content

            elif isinstance(content, list):

                for block in content:

                    if isinstance(block, dict):

                        if block.get("type") == "text":

                            text = block.get("text", "")

                            if text:
                                yield text


# ============================================================
# 8. SAISIE DE LA QUESTION PAR L'UTILISATEUR
# ============================================================

if agent_available:

    prompt = st.chat_input(
        "Posez votre question à Datheon AI..."
    )

    if prompt:

        # Ajouter la question à l'historique
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # Afficher la question de l'utilisateur
        with st.chat_message("user"):

            st.markdown(prompt)

        # Générer la réponse par notre Assistant
        with st.chat_message("assistant"):

            try:

                with st.spinner("L'agent analyse votre demande..."):

                    response = st.write_stream(
                        stream_agent_response()
                    )

                if not response:

                    response = (
                        "Je n'ai pas pu générer de réponse."
                    )

                    st.warning(response)

                # Enregistrer la réponse dans l'historique
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            except Exception as error:

                error_message = (
                    f"Une erreur est survenue : {error}"
                )

                st.error(error_message)

                # Retirer la dernière question pour éviter
                # de conserver un échange incomplet.
                st.session_state.messages.pop()

else:

    st.warning(
        "Veuillez corriger les erreurs de configuration "
        "de votre agent avant de démarrer la conversation."
    )


# ============================================================
# 9. PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Datheon AI Assistant | Powered by Google Gemini "
    "and LangChain"
)
