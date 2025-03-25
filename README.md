# Raggle_Assistant
Below is a refined and expanded version of the `README.md` file, incorporating all your requested inclusions and providing in-depth explanations suitable for a company project documentation. This version is detailed, professional, and tailored to showcase the technical prowess and reasoning behind every component of Raggle.

---

# Raggle - Retrieval-Augmented Generation Chatbot

![Raggle Logo](https://img.shields.io/badge/Raggle-Chatbot-green)  
**A scalable, multi-modal RAG-based chatbot with a well-designed UI/UX for querying documents, images, videos, and YouTube content.**

---

## Problem Statement

With the explosion of multi-modal data (documents, images, videos, and online media), users face challenges in efficiently extracting actionable insights. Traditional systems lack the ability to process diverse inputs cohesively, retrieve relevant context, and generate accurate, data-grounded responses. Raggle addresses this by providing a scalable, user-friendly solution that leverages Retrieval-Augmented Generation (RAG) to process and query multi-modal content effectively.

---

## RAG Model Architecture

Raggle employs a **Retrieval-Augmented Generation (RAG)** architecture, optimized for scalability and performance:

1. **Input Processing**: Multi-modal inputs (PDFs, DOCX, CSVs, TXTs, images, videos, YouTube URLs) are processed into text chunks using specialized libraries and models (e.g., OCR, audio transcription).
2. **Embedding Generation**: Text chunks are transformed into dense vector representations using the `SentenceTransformer` model (`all-MiniLM-L6-v2`).
3. **Vector Storage**: Embeddings are indexed in FAISS for fast retrieval and persisted in MongoDB for durability.
4. **Retrieval**: User queries are embedded and matched against stored vectors to retrieve the top-3 relevant chunks.
5. **Generation**: Retrieved contexts are passed to the Gemini 2.0 Flash model via its API to generate concise, context-aware responses.

### Why Flask Backend?
We chose **Flask** as the backend framework for its lightweight nature and ability to scale horizontally. Flask’s RESTful API design enables seamless integration with the Streamlit frontend, efficient handling of file uploads, and concurrent query processing, making it ideal for enterprise-level scalability.

---

## Objectives

- **Multi-Modal Processing**: Support a wide range of inputs (PDF, DOCX, CSV, TXT, PNG, JPG, MP4, YouTube URLs).
- **Scalable Retrieval**: Ensure fast and accurate retrieval of relevant data using a hybrid FAISS-MongoDB storage system.
- **Contextual Accuracy**: Deliver responses grounded in user-uploaded content using RAG and Gemini.
- **Superior UI/UX**: Provide a well-designed, intuitive Streamlit interface for easy interaction.
- **Data Security**: Ensure user data privacy by clearing the database upon session end.

---

## How to Set Up the Project

### Prerequisites
- **OS**: Windows, macOS, or Linux
- **Python**: 3.9+
- **MongoDB**: Running locally on `localhost:27017`
- **FFmpeg**: Installed for audio/video processing
- **Git**: For repository cloning

### Setup Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/raggle.git
   cd raggle
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   See [Technical Stack](#technical-stack) for the full list.

4. **Configure Environment Variables**
   Create a `.env` file:
   ```plaintext
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   Obtain from [Google's Gemini API](https://ai.google.dev/).

5. **Download CLIP Model**
   ```bash
   python download_clip_model.py
   ```
   Saves to `d:/RAG_Chatbot/models/clip-vit-base-patch16` (adjust path as needed).

6. **Start MongoDB**
   ```bash
   mongod
   ```

7. **Run the Application**
   ```bash
   python app.py
   ```
   - Flask: `http://localhost:5000`
   - Streamlit: `http://localhost:8501`

8. **Access the UI**
   Visit `http://localhost:8501`.

---

## Technical Stack

- **Backend**: Flask (scalable REST API)
- **Frontend**: Streamlit (intuitive UI/UX)
- **Embedding Model**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS (fast similarity search)
- **Persistent Storage**: MongoDB (data durability)
- **Generative Model**: Gemini 2.0 Flash (via API)
- **File Processing**: PyPDF2, python-docx, pandas, OpenCV, EasyOCR, Whisper, youtube_transcript_api, yt_dlp
- **Audio/Video**: ffmpeg-python
- **Utilities**: pytesseract, pyspellchecker, sentence-transformers

---

## Methodology

### Ideation
Raggle was conceived to address the gap in multi-modal data querying, prioritizing scalability, user experience, and data security. The use of Flask ensures scalability, while Streamlit provides a polished UI/UX.

### Implementation
1. **File Processing**: Custom handlers extract text from various formats.
2. **Embedding**: SentenceTransformer generates vectors.
3. **Storage**: FAISS indexes vectors, MongoDB stores metadata.
4. **Retrieval & Generation**: FAISS retrieves contexts, Gemini generates responses.
5. **UI**: Streamlit delivers a seamless user experience.

---

## Workflow

1. **Upload**: Users upload files or YouTube URLs via the sidebar.
2. **Processing**: Content is chunked, embedded, and stored.
3. **Query**: Users ask questions in the chat interface.
4. **Retrieval**: Top-3 relevant chunks are fetched.
5. **Response**: Gemini generates a formatted answer.
6. **Display**: Responses appear in the chat with timestamps.

---

## Detailed Processing and Storage

### Document Processing (DOCX, PDF, TXT, CSV)
- **DOCX**: `python-docx` extracts paragraph text, joined with newlines.
  - *Why*: Simple, reliable parsing of Word documents.
- **PDF**: `PyPDF2` reads pages, extracts text with encryption support via `pycryptodome`.
  - *Why*: Robust handling of PDFs, including encrypted ones.
- **TXT**: Decoded directly from file bytes.
  - *Why*: Minimal overhead for plain text.
- **CSV**: `pandas` reads rows, chunked into 100-row segments, converted to string.
  - *Why*: Structured data preserved for querying.

**Chunking**: Text is split into 1000-character chunks with 200-character overlap to maintain context.

**Storage in FAISS and MongoDB**:
- **FAISS**: Vectors are added to a `FlatL2` index for fast similarity search.
  - *Why*: FAISS offers sub-linear search time, ideal for large datasets.
- **MongoDB**: Stores vectors, text, and metadata (e.g., chunk size, source).
  - *Why*: Ensures persistence and allows querying beyond memory constraints.

**Example**:
- Input: "This is a sample PDF document about AI."
- Chunk: "This is a sample PDF docu" (ID: md5 hash)
- Vector: `embedder.encode(chunk)` → `[0.12, -0.45, ...]` (384 dims)
- FAISS: Adds vector to index.
- MongoDB: Stores `{"vector_id": 0, "vector": [0.12, -0.45, ...], "text": "This is...", "metadata": {"size": 25, "type": "pdf"}}`.

### SentenceTransformer (`all-MiniLM-L6-v2`) Architecture
- **Model**: A transformer-based model fine-tuned on sentence embeddings.
- **Layers**: 6 transformer layers, 384-dimensional output.
- **Architecture**: MiniLM (distilled BERT), optimized for speed and efficiency.
- **Why**: Balances performance and resource usage, suitable for real-time embedding.

---

### Image Processing
- **Techniques**:
  - **OpenCV**: Preprocesses images (grayscale, bilateral filtering, thresholding) to enhance OCR.
  - **EasyOCR**: Extracts text from multiple preprocessed versions.
    - *Model*: CNN + RNN with language-specific training (English).
    - *Why*: Robust text detection across varied image qualities.
  - **Spellchecker**: `pyspellchecker` corrects OCR errors.
    - *Why*: Improves accuracy of extracted text (e.g., "teh" → "the").
- **Process**: Image → Multiple enhancements → OCR → Spell correction → Chunking → Embedding.
- **Why**: Ensures reliable text extraction from noisy or low-quality images.

---

### Video Processing
- **Techniques**:
  - **Audio Extraction**: `ffmpeg-python` extracts WAV audio from MP4/AVI.
    - *Why*: FFmpeg is industry-standard, fast, and reliable for audio processing.
  - **Transcription**: `Whisper` (base model) transcribes audio to text.
    - *Model*: Transformer-based, trained on multilingual audio.
    - *Why*: High accuracy, supports translation.
  - **Frame Text**: OpenCV detects scene changes (frame diff > 30), EasyOCR extracts text.
    - *Spellchecker*: Corrects OCR errors.
- **Process**: Video → Audio extraction → Transcription → Frame text extraction → Chunking → Embedding.
- **Why**: Captures both audio and visual content for comprehensive querying.

---

### YouTube Video Processing
- **Video ID Extraction**: Regex extracts 11-character ID from URLs (e.g., `v=abc123`).
  - *Why*: Handles various URL formats (youtu.be, youtube.com).
- **Audio Extraction**: `yt_dlp` downloads audio as WAV.
  - *Why*: Robust, supports latest YouTube formats.
- **Transcript**: `youtube_transcript_api` fetches if available; else, Whisper transcribes audio.
- **Embedding**: Chunks (summary, transcript, description) are embedded and stored.
- **Why**: Extends functionality to online content, enhancing versatility.

---

## Frontend Information
- **Sidebar**: Upload options, processed document list, database stats, model status.
- **Chat**: Real-time query/response display with timestamps.
- **Why**: Ensures easy access to all features, enhancing UX.

---

## Data Security
- **Policy**: User data is not stored persistently. The database (FAISS + MongoDB) is cleared when the user exits the session.
- **Why**: Protects privacy, complies with data security standards.

---

## Innovations
- **Scalable Backend**: Flask enables horizontal scaling for enterprise use.
- **UI/UX**: Streamlit’s polished design ensures accessibility and engagement.
- **Hybrid Storage**: FAISS + MongoDB balances speed and persistence.
- **Multi-Modal Robustness**: OCR, Whisper, and YouTube integration cover all bases.
- **Error Correction**: Spellchecker enhances text quality from images/videos.

---

## Output
- **UI**: Streamlit app with chat and upload features.
- **Responses**: Context-aware answers (e.g., "The PDF says AI improves efficiency...").
- **Stats**: Vector count, storage type displayed in real-time.

---

## Contributing
Fork, branch, commit, and submit a PR. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License
MIT License - see [LICENSE](LICENSE).

---

## Acknowledgments
- Developed by **Jagdeesh P** for enterprise-grade RAG solutions.
- Inspired by xAI’s mission to advance AI innovation.

*Version: 1.2.5 | Updated: March 24, 2025*

---

### Additional Notes:
- Ensure `requirements.txt` is included with all dependencies.
- Adjust file paths (e.g., `MODEL_DIR`) based on your environment.
- This README is now a thorough company-grade document, covering all aspects of Raggle comprehensively.

Let me know if further refinements are needed!
