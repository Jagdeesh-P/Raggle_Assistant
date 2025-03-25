import requests
import json
import os
from dotenv import load_dotenv
import importlib.util

# Load environment variables from .env file
load_dotenv()

# Get environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class LLMHandler:
    def __init__(self, model_type="gemini"):
        """
        Initialize LLM handler with model choice.
        model_type options: "gemini"
        """

        model_type = "gemini"
        self.model_type = model_type

    def generate_prompt(self, query, contexts):
        """Generate a well-structured prompt with context for the LLM."""
        context_text = "\n\n".join([f"{ctx['metadata'].get('type', 'UNKNOWN').upper()}:\n{ctx['text']}" for ctx in contexts])
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided document contexts.
        
        CONTEXT INFORMATION:
        {context_text}
        
        INSTRUCTIONS:
        - Try to understand the context of the data and context of the user query completely first.
        - Answer the user's question based ONLY on the provided context
        - If the answer cannot be determined from the context, say "I don't have enough information to answer that question based on the documents provided."
        - Do not make up information or use knowledge outside the provided context but try to extract the info or form the info from the given data.
        - Respond in a clear, concise, and helpful manner
        - If the context contains tables or structured data, format your response appropriately
        - Cite specific parts of the documents if available only.
        - Ensure your response is accurate and relevant to the user's question
        - If user asks for summarize, just understand the context of data and summarize correctly.
        
        USER QUESTION: {query}
        """
        return prompt

    def generate_response(self, query, contexts):
        """Generate a response based on query and retrieved contexts."""
        prompt = self.generate_prompt(query, contexts)

        if self.model_type == "gemini":
            return self._call_gemini_api(prompt)
        else:
            return "Model type not configured properly."

    def _call_gemini_api(self, prompt):
        """Call Gemini API to generate a response."""
        if not GEMINI_API_KEY:
            return "Gemini API key not set. Please set the GEMINI_API_KEY environment variable."

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,  # Lower temperature for more focused responses
                "topK": 40,          # Reduced from 100 to focus on more relevant tokens
                "topP": 0.9,         # Slightly reduced for more deterministic responses
                "maxOutputTokens": 8192,  # Increased to allow for longer responses
                "stopSequences": [],
                "candidateCount": 1,
                "presencePenalty": 0.6,  # Add presence penalty to encourage consideration of all context
                "frequencyPenalty": 0.8,  # Add frequency penalty to discourage repetition
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response_json = response.json()

            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                answer = response_json["candidates"][0]["content"]["parts"][0]["text"]
                return answer
            else:
                return "Error generating response from Gemini API."
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"
