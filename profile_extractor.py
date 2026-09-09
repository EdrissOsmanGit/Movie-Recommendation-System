# profile_extractor.py
EXTRACT_PROMPT = """From this conversation, extract the user's movie preferences as JSON:
{
  "genres": ["Action", "Thriller"],
  "mood": "dark and intense",
  "similar_to": ["The Dark Knight"],
  "avoid": ["gore", "slow pacing"],
  "era": "modern"
}
Return only JSON."""

def extract_profile(conversation: str, model="llama3.2:3b") -> dict:
    response = call_ollama(model, EXTRACT_PROMPT, conversation)
    return json.loads(response)