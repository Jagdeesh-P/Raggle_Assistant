import os
import json
import time
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from file_handler import process_file, search_documents, get_collection_stats, clear_database, initialize_index
from llm import LLMHandler
from io import BytesIO
from functools import lru_cache
import re  # Added import for regular expressions

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize LLM handler
llm_handler = LLMHandler(model_type="gemini")

# Initialize FAISS index and database on startup
clear_database()
initialize_index()

# Store documents in memory
documents = []

# Helper class to mimic Streamlit's UploadedFile interface
class FileWrapper:
    def __init__(self, file_storage):
        self.name = file_storage.filename
        self.type = file_storage.content_type
        self._file = file_storage
        self._bytes = None
        self._position = 0

    def getvalue(self):
        if self._bytes is None:
            self._bytes = self._file.read()
            self._file.seek(0)
        return self._bytes

    def read(self, size=None):
        if self._bytes is None:
            self._bytes = self._file.read()
            self._file.seek(0)
        if size is None:
            result = self._bytes[self._position:]
            self._position = len(self._bytes)
            return result
        else:
            end = min(self._position + size, len(self._bytes))
            result = self._bytes[self._position:end]
            self._position = end
            return result

    def seek(self, offset, whence=0):
        if self._bytes is None:
            self._bytes = self._file.read()
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        elif whence == 2:
            self._position = len(self._bytes) + offset
        self._position = max(0, min(self._position, len(self._bytes)))

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global documents
    if 'file' in request.files:
        uploaded_file = request.files['file']
        file_extension = uploaded_file.filename.split(".")[-1].lower()
        wrapped_file = FileWrapper(uploaded_file)
        try:
            result = process_file(wrapped_file, file_extension)
            if "error" in result:
                return jsonify({"error": result["error"]}), 400
            if uploaded_file.filename not in [doc["name"] for doc in documents]:
                documents.append({
                    "name": uploaded_file.filename,
                    "type": file_extension,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            return jsonify({
                "message": f"Successfully processed {uploaded_file.filename} and added {result['chunks_added']} chunks",
                "documents": documents
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    elif 'youtube_url' in request.form:
        youtube_url = request.form['youtube_url']
        try:
            result = process_file(youtube_url, "youtube")
            if "error" in result:
                return jsonify({"error": result["error"]}), 400
            video_name = "YouTube Video"
            video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
            if video_id:
                video_name = f"YouTube: {video_id.group(1)}"
            if video_name not in [doc["name"] for doc in documents]:
                documents.append({
                    "name": video_name,
                    "type": "youtube",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "url": youtube_url
                })
            return jsonify({
                "message": f"Successfully processed YouTube video and added {result['chunks_added']} chunks",
                "documents": documents
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "No file or YouTube URL provided"}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    if not documents:
        return jsonify({"response": "Please upload and process at least one document first."}), 200
    contexts = search_documents(user_query, n_results=3)
    response = llm_handler.generate_response(user_query, contexts)
    return jsonify({
        "response": response,
        "timestamp": time.strftime("%H:%M:%S")
    }), 200

@lru_cache(maxsize=1)
def cached_documents():
    return documents

@app.route('/api/documents', methods=['GET'])
def get_documents():
    return jsonify({"documents": cached_documents()}), 200

@lru_cache(maxsize=1)
def cached_stats():
    return get_collection_stats()

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(cached_stats()), 200

@app.route('/api/clear', methods=['POST'])
def clear_db():
    global documents
    clear_database()
    documents = []
    cached_documents.cache_clear()  # Clear cache
    cached_stats.cache_clear()  # Clear cache
    return jsonify({"message": "Database cleared successfully"}), 200

@lru_cache(maxsize=1)
def cached_config():
    return {"gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY"))}

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(cached_config()), 200

def launch_streamlit_app():
    subprocess.Popen(["streamlit", "run", "streamlit_app.py", "--server.port", "8501"])

if __name__ == '__main__':
    print("Starting Flask backend on port 5000 and launching Streamlit app on port 8501...")
    launch_streamlit_app()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)