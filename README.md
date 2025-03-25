# Raggle - Retrieval-Augmented Generation Chatbot

![Raggle Logo](https://img.shields.io/badge/Raggle-Chatbot-green)  
**A custom RAG-based chatbot for querying multi-modal documents without LangChain.**


## Problem Statement

The goal is to develop a Retrieval-Augmented Generation (RAG)-based chatbot capable of answering user queries by retrieving and processing information from diverse document types: PDFs, DOCX, CSVs, images, and videos. The chatbot must extract relevant data accurately, avoid reliance on the LangChain framework, and provide contextually appropriate responses. Sample files for PDFs, DOCX, and CSVs are provided, while image and video datasets are self-selected to demonstrate custom processing logic.

---

## Objectives

- **Multi-Format Support**: Process PDFs, DOCX, CSVs (provided samples), and self-chosen image/video datasets.
- **Accurate Retrieval**: Retrieve precise, query-relevant information from stored documents.
- **Contextual Responses**: Generate answers based solely on document content using RAG.
- **Custom Implementation**: Avoid LangChain, showcasing a bespoke RAG architecture.
- **Optimization**: Employ efficient techniques for performance and scalability.

---

## RAG Model Architecture

Raggle implements a custom RAG pipeline optimized for multi-modal data:

1. **Input Processing**: 
   - PDFs, DOCX, CSVs: Extracted using PyPDF2, python-docx, and pandas.
   - Images: OCR with OpenCV and EasyOCR.
   - Videos: Audio transcription (Whisper) and frame text (EasyOCR).
2. **Text Chunking**: Documents split into 1000-character chunks with 200-character overlap.
3. **Embedding**: `SentenceTransformer` (`all-MiniLM-L6-v2`) generates 384-dimensional vectors.
4. **Storage**: 
   - FAISS (`IndexFlatL2`) for fast similarity search.
   - MongoDB for persistent metadata and vector storage.
5. **Retrieval**: Top-k relevant chunks retrieved via FAISS based on query embedding.
6. **Generation**: We utilize the Gemini 2.0 Flash API from Google as our LLM to generate responses from retrieved contexts. Gemini 2.0 Flash was chosen for its exceptional balance of speed, cost-efficiency, and contextual accuracy, making it ideal for real-time chatbot applications.

- **CLIP Integration**: The CLIP model (`clip-vit-base-patch16`) from OpenAI is incorporated to potentially enhance multi-modal processing. We chose CLIP for its ability to generate joint embeddings for images and text, enabling future capabilities like semantic image/video frame retrieval (e.g., querying visual content directly without relying solely on extracted text). Although currently loaded in video processing, its full potential is reserved for future enhancements, such as embedding raw frames for richer context beyond OCR-based text extraction.

*Why No LangChain?*  
We built a lightweight, custom RAG pipeline to maintain full control over processing, storage, and retrieval logic, ensuring transparency and optimization tailored to the task.

---

## How to Set Up the Project

### Prerequisites
- **OS**: Windows, macOS, or Linux
- **Python**: 3.9+
- **MongoDB**: Running on `localhost:27017`
- **FFmpeg**: For video/audio processing
- **Git**: For cloning

### Setup Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Jagdeesh-P/Raggle_Assistant.git
   cd Raggle_Assistant
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
   See [Technical Stack](#technical-stack) for details.

4. **Configure Environment**
   Create `.env`:
   ```plaintext
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   Obtain from [Google's Gemini API](https://ai.google.dev/).

5. **Download CLIP Model**
   ```bash
   python download_model.py
   ```
   Adjust `MODEL_DIR` if needed (`d:/RAG_Chatbot/models/clip-vit-base-patch16`).

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

8. **Access UI**
   Visit `http://localhost:8501`.

---

## Technical Stack

- **Backend**: Flask (REST API for scalability)
- **Frontend**: Streamlit (intuitive UI)
- **Embedding**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS (fast retrieval)
- **Storage**: MongoDB (persistence)
- **Generation**: Gemini 2.0 Flash (API)
- **Processing**: 
  - PDFs: PyPDF2
  - DOCX: python-docx
  - CSVs: pandas
  - Images: OpenCV, EasyOCR
  - Videos: ffmpeg-python, Whisper, OpenCV, EasyOCR
- **Utilities**: pyspellchecker, youtube_transcript_api, yt_dlp

---

## Document Processing

### PDFs, DOCX, CSVs
- **PDF**: `PyPDF2` extracts text from pages, supports encrypted files with `pycryptodome`.
  - *Example*: "AI is transformative" → Chunked → Embedded.
- **DOCX**: `python-docx` retrieves paragraph text.
  - *Example*: "Machine learning advances" → Joined → Chunked.
- **CSV**: `pandas` reads rows, chunks every 100 rows into strings.
  - *Example*: 100 rows of sales data → "date,amount\n2023,500" → Embedded.

**Chunking**: 1000-char chunks with overlap for context preservation.

**Storage**:
- **FAISS**: Vectors stored in `IndexFlatL2` for L2 distance search.
- **MongoDB**: Metadata (e.g., `{"type": "pdf", "size": 25}`) and vectors persisted.
- *Why*: FAISS ensures speed; MongoDB ensures durability.

**Example**:
- Text: "AI improves efficiency."
- Chunk: "AI improves effic"
- Vector: `[0.1, -0.3, ...]` (384 dims)
- FAISS: Index entry
- MongoDB: `{"vector_id": 1, "text": "AI improves effic", "vector": [0.1, -0.3, ...], "metadata": {"type": "pdf"}}`

---

### Image Processing
- **Dataset**: Self-selected (e.g., scanned documents, signs).
- **Techniques**:
  - **OpenCV**: Grayscale, bilateral filtering, adaptive thresholding enhance text visibility.
  - **EasyOCR**: CNN+RNN model extracts text from multiple processed versions.
    - *Why*: Robust across image quality variations.
  - **Spellchecker**: `pyspellchecker` corrects OCR errors (e.g., "machin" → "machine").
    - *Why*: Boosts accuracy of extracted text.
- **Process**: Image → Enhancements → OCR → Correction → Chunking → Embedding.

---

### Video Processing
- **Dataset**: Self-selected (e.g., tutorial videos, presentations).
- **Techniques**:
  - **Audio**: `ffmpeg-python` extracts WAV audio.
    - *Why*: FFmpeg is fast and reliable for media processing.
  - **Transcription**: `Whisper` (base model, Transformer-based) transcribes audio.
    - *Why*: High accuracy, multilingual support.
  - **Frame Text**: OpenCV detects scene changes (frame diff > 30), EasyOCR extracts text.
    - *Spellchecker*: Corrects OCR errors.
- **Process**: Video → Audio extraction → Transcription → Frame text → Chunking → Embedding.
- **Why CLIP?**: We included the CLIP model to lay the groundwork for advanced video analysis. CLIP’s pre-trained vision-language capabilities allow us to potentially embed video frames directly, offering a pathway to answer queries like "Find scenes with a whiteboard" without needing text extraction. While the current implementation relies on EasyOCR and Whisper for text-based retrieval, CLIP’s inclusion reflects our intent to optimize for visual semantics in future iterations, leveraging its zero-shot classification and image-text alignment strengths.

---

## Evaluation Criteria Fulfillment

### Accuracy & Relevance
- Custom RAG ensures responses are grounded in document content, avoiding hallucination.
- Top-k retrieval with FAISS guarantees relevant context.

### Model Architecture
- Lightweight, modular design with Flask backend and custom FAISS-MongoDB integration.
- No LangChain dependency showcases bespoke optimization.

### Handling Different Formats
- Tailored parsing for PDFs, DOCX, CSVs.
- Creative image/video processing with OCR and audio transcription.

### Creativity & Optimization
- Scene detection in videos for smarter frame sampling.
- Spellchecker improves text quality.
- Hybrid storage balances speed and persistence.

---

## Workflow
1. **Upload**: Files via Streamlit sidebar.
2. **Process**: Extracted, chunked, embedded, stored.
3. **Query**: User asks (e.g., "What’s in the PDF?").
4. **Retrieve**: FAISS fetches top-k chunks.
5. **Generate**: Gemini crafts response.
6. **Display**: Answer shown with timestamp.

---

## Output
- **UI**: Streamlit interface with upload and chat features.
- **Responses**: E.g., "The PDF states AI improves efficiency in task automation."
- **Stats**: Vector count displayed for transparency.

---

## License
MIT License - see [LICENSE](LICENSE).

---

## Acknowledgments
- Built by **Jagdeesh P** for a custom RAG solution.
- Sample files provided; image/video datasets self-curated.

*Version: 1.2.5 | Updated: March 24, 2025*

---
