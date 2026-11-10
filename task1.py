
import os
from typing import List, Optional
import time, random 
from google.genai import errors 

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Charger les variables d'environnement
load_dotenv()

# 2. Vérifier la clé API
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("La variable GEMINI_API_KEY est absente du fichier .env")

# 3. Définir les structures Pydantic
class Experience(BaseModel):
    title: str = Field(description="Intitulé du poste")
    company: str = Field(description="Nom de l'entreprise")
    duration: Optional[str] = Field(
        default=None,
        description="Durée ou période de l'expérience"
    )

class CandidateProfile(BaseModel):
    full_name: str = Field(description="Nom complet du candidat")
    email: Optional[str] = Field(
        default=None,
        description="Adresse email"
    )
    skills: List[str] = Field(
        description="Liste des compétences techniques clés"
    )
    experiences: List[Experience] = Field(
        description="Liste des expériences professionnelles"
    )

# 4. Initialiser le client Gemini
client = genai.Client(api_key=api_key)

# 5. Texte brut à analyser
raw_text = """
Azael BIVIHOU est un Data & AI Engineer basé à Tunis.
Contact: azaelbivihou53@gmail.com.

Compétences: Python, SQL, LLMs, LangChain, RAG,
Multi-Agent Systems, Pydantic.

Expériences:
- Data Science and AI Intern chez FARNESS UAV SOFTWARE,
où il a conçu un système de coordination de drones autonomes.
- Web Developer chez PRODECOM,
développement d'applications web de gestion de garage.
"""

# 6. Fonction d'extraction des données structurées
def extract_structured_data(text: str) -> CandidateProfile:

    prompt = f"""
    Tu es un expert en extraction de données de CV.

    Extrais uniquement les informations présentes dans le texte.
    N'invente aucune information.
    Si une information est absente, utilise null lorsque le
    champ est optionnel ou une liste vide lorsque nécessaire.

    Texte à analyser :
    {text}
    """

    max_retries = 5
    for attempt in range(max_retries) :
        try :

            print(f"Tentative {attempt + 1}/{max_retries}...")

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CandidateProfile,
                    temperature=0.1,
                ),
            )

            if response.parsed is not None :
                return response.parsed

            return CandidateProfile.model_validate_json(response.text)

        except errors.APIError as e :

            #
            if e.code in (408, 429, 500, 502, 503, 504):
                if attempt == max_retries -1 :
                    raise
                delay = min(2 ** (attempt + 1), 30) + random.uniform(0, 1)
                print(f"Serveur indisponible ({e.code}).")
                print(f"Nouvelle tentative dans {delay:.1f} secondes...")

                time.sleep(delay)

            else:
                raise    

    # Validation avec Pydantic
    if response.parsed is not None:
        return response.parsed

    return CandidateProfile.model_validate_json(response.text)


# 7. Exécution principale
if __name__ == "__main__":
    try:
        result = extract_structured_data(raw_text)

        print("=== DONNÉES STRUCTURÉES ET VALIDÉES EN JSON ===")

        print(result.model_dump_json(indent=2))

    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")
