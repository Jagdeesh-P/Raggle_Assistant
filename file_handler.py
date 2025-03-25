import os
import tempfile
from typing import List, Dict, Any
import hashlib
import numpy as np
import pandas as pd
import docx
import PyPDF2
import cv2
import easyocr
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
from pymongo import MongoClient
import logging
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
import yt_dlp
import re
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker

# Initialize spell checker with default English corpus
spell = SpellChecker()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Model directory for CLIP
MODEL_DIR = "d:/RAG_Chatbot/models/clip-vit-base-patch16"

# Define base path for database storage
DB_DIR = "d:/RAG_Chatbot/vector_db"
os.makedirs(DB_DIR, exist_ok=True)

# MongoDB setup
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['rag_chatbot']
docs_collection = db['documents']

# Whisper model initialization (loaded globally with fallback)
WHISPER_MODEL = None
try:
    import whisper
    WHISPER_MODEL = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import whisper: {str(e)}. Audio transcription will be skipped.")
except Exception as e:
    logger.warning(f"Could not load Whisper model: {str(e)}. Falling back to transcript API only.")

# FAISS setup with MongoDB persistence
class FAISS_VectorDB:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_metadata = {}
        self.load_from_mongodb()

    def load_from_mongodb(self):
        try:
            all_docs = docs_collection.find()
            vectors_to_add = []
            for doc in all_docs:
                vector_id = doc['vector_id']
                vector = np.array(doc['vector'], dtype=np.float32)
                text = doc['text']
                metadata = doc['metadata']
                self.id_to_metadata[vector_id] = {"text": text, "metadata": metadata}
                vectors_to_add.append(vector)
            if vectors_to_add:
                self.index.add(np.array(vectors_to_add))
                logger.info(f"Loaded {len(vectors_to_add)} vectors from MongoDB into FAISS")
        except Exception as e:
            logger.error(f"Error loading from MongoDB: {str(e)}")

    def add_vectors(self, vectors: List[np.ndarray], texts: List[str], metadatas: List[Dict]):
        start_id = self.index.ntotal
        self.index.add(np.array(vectors, dtype=np.float32))
        for vec_id, vector, text, metadata in zip(range(start_id, start_id + len(vectors)), vectors, texts, metadatas):
            self.id_to_metadata[vec_id] = {"text": text, "metadata": metadata}
            doc = {"vector_id": vec_id, "vector": vector.tolist(), "text": text, "metadata": metadata}
            try:
                docs_collection.insert_one(doc)
            except Exception as e:
                logger.error(f"Error saving to MongoDB: {str(e)}")

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx in self.id_to_metadata:
                doc_data = self.id_to_metadata[int(idx)]
                results.append({"text": doc_data["text"], "metadata": doc_data["metadata"], "distance": float(distance)})
            else:
                doc = docs_collection.find_one({"vector_id": int(idx)})
                if doc:
                    results.append({"text": doc["text"], "metadata": doc["metadata"], "distance": float(distance)})
                    self.id_to_metadata[int(idx)] = {"text": doc["text"], "metadata": doc["metadata"]}
        return results

# Global FAISS instance
dimension = 384
faiss_db = FAISS_VectorDB(dimension)

# Embedding function (simple MD5-based for now)

def correct_spelling(text: str) -> str:
    """Correct spelling in OCR-extracted text using pyspellchecker."""
    words = text.split()
    corrected_words = []
    for word in words:
        if spell.unknown([word]):  # If word is not in dictionary
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)  # Fallback to original if no correction
        else:
            corrected_words.append(word)  # Keep correct words as-is
    corrected_text = " ".join(corrected_words)
    if corrected_text != text:
        logger.info(f"Corrected text: '{text}' -> '{corrected_text}'")
    return corrected_text

def embed_text(text: str) -> np.ndarray:
    return embedder.encode(text, convert_to_numpy=True)

def generate_chunk_id(text: str) -> str:
    normalized_text = text.strip().lower()  # Basic normalization
    return hashlib.md5(normalized_text.encode()).hexdigest()

def process_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    chunks = []
    if len(text) <= chunk_size:
        chunk_id = generate_chunk_id(text)
        chunks.append({"id": chunk_id, "text": text, "metadata": {"size": len(text)}})
        return chunks
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if len(chunk_text) < 50:
            continue
        chunk_id = generate_chunk_id(chunk_text)
        chunks.append({"id": chunk_id, "text": chunk_text, "metadata": {"size": len(chunk_text), "start_char": i}})
    return chunks

def extract_video_id(youtube_url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    pattern = r"(?:https?://)?(?:www\.)?(?:(?:youtube\.com/watch\?v=)|(?:youtu\.be/)|(?:youtube\.com/embed/)|(?:youtube\.com/v/)|(?:youtube\.com/shorts/))([a-zA-Z0-9_-]{11})(?:\?.*|\S*)?$"
    match = re.search(pattern, youtube_url)
    if match:
        logger.info(f"Extracted video ID: {match.group(1)} from URL: {youtube_url}")
        return match.group(1)
    logger.error(f"No valid video ID found in URL: {youtube_url}")
    return None

def download_audio(video_id: str) -> str:
    """Download audio from YouTube video using yt_dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'wav',
        'outtmpl': os.path.join(tempfile.gettempdir(), f'{video_id}.%(ext)s'),
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        audio_path = os.path.join(tempfile.gettempdir(), f"{video_id}.wav")
        if os.path.exists(audio_path):
            logger.info(f"Audio downloaded successfully: {audio_path}")
            return audio_path
        raise FileNotFoundError("Audio file not created")
    except Exception as e:
        logger.error(f"Error downloading audio for video ID {video_id}: {str(e)}")
        return None

def audio_to_text(audio_path: str) -> str:
    """Transcribe audio to text using Whisper."""
    if not WHISPER_MODEL:
        logger.warning("Whisper model not available; skipping audio transcription")
        return None
    try:
        result = WHISPER_MODEL.transcribe(audio_path, task="translate")
        text = result['text']
        logger.info(f"Audio transcribed successfully: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return None
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            logger.info(f"Cleaned up audio file: {audio_path}")

def process_youtube(youtube_url: str) -> List[Dict[str, Any]]:
    """Process a YouTube video URL and return text chunks for vectorization."""
    # Validate and extract video ID
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return [{"id": "error", "text": f"Invalid YouTube URL: {youtube_url}", "metadata": {"error": "invalid_youtube_url"}}]
    
    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    chunks = []

    # Fetch metadata
    try:
        youtube = YouTube(clean_url, use_oauth=False, allow_oauth_cache=False)
        video_title = youtube.title or f"YouTube Video {video_id}"
        video_author = youtube.author or "Unknown"
        video_length = youtube.length or 0
        video_description = youtube.description or ""
        logger.info(f"Metadata fetched: {video_title} by {video_author}")
    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {video_id}: {str(e)}")
        video_title, video_author, video_length, video_description = f"YouTube Video {video_id}", "Unknown", 0, ""

    # Add summary chunk
    summary_text = f"YouTube Video: {video_title}\nAuthor: {video_author}\nLength: {video_length} seconds\nDescription: {video_description}"
    chunks.append({
        "id": generate_chunk_id(summary_text),
        "text": summary_text,
        "metadata": {"type": "youtube_summary", "title": video_title, "author": video_author, "length": video_length, "url": clean_url, "video_id": video_id}
    })

    # Attempt transcript retrieval
    transcript_text = None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        transcript_text = " ".join([entry['text'] for entry in transcript])
        logger.info(f"Transcript retrieved for {video_id}: {len(transcript_text)} characters")
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.info(f"No transcript available for {video_id}; attempting audio fallback")
    except Exception as e:
        logger.error(f"Error fetching transcript for {video_id}: {str(e)}")

    # Fallback to audio transcription if no transcript
    if not transcript_text and WHISPER_MODEL:
        audio_path = download_audio(video_id)
        if audio_path:
            transcript_text = audio_to_text(audio_path)
            if transcript_text:
                logger.info(f"Audio transcription successful for {video_id}")
            else:
                logger.warning(f"Audio transcription failed for {video_id}")

    # Process content
    if transcript_text:
        transcript_chunks = process_text(transcript_text)
        for chunk in transcript_chunks:
            chunk["metadata"].update({
                "type": "youtube_transcript",
                "title": video_title,
                "url": clean_url,
                "video_id": video_id
            })
        chunks.extend(transcript_chunks)
    elif video_description:
        description_chunks = process_text(video_description)
        for chunk in description_chunks:
            chunk["metadata"].update({
                "type": "youtube_description",
                "title": video_title,
                "url": clean_url,
                "video_id": video_id
            })
        chunks.extend(description_chunks)
    else:
        chunks.append({
            "id": generate_chunk_id(f"No content for {video_id}"),
            "text": f"No transcript or description available for this YouTube video (ID: {video_id})",
            "metadata": {"type": "youtube_error", "error": "no_content", "video_id": video_id}
        })

    return chunks

# Other processing functions (unchanged)
def process_pdf(file) -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
        temp.write(file.getvalue())
        temp_path = temp.name
    text = ""
    try:
        try:
            from Crypto.Cipher import AES
            has_pycryptodome = True
        except ImportError:
            has_pycryptodome = False
        with open(temp_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            if pdf_reader.is_encrypted:
                if not has_pycryptodome:
                    os.unlink(temp_path)
                    return [{"id": "error", "text": "This PDF is encrypted and requires PyCryptodome.", "metadata": {"error": "missing_dependency"}}]
                try:
                    pdf_reader.decrypt('')
                except:
                    os.unlink(temp_path)
                    return [{"id": "error", "text": "This PDF is encrypted and cannot be processed.", "metadata": {"error": "encrypted_pdf"}}]
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        os.unlink(temp_path)
        return [{"id": "error", "text": f"Error processing PDF: {str(e)}", "metadata": {"error": "pdf_processing"}}]
    os.unlink(temp_path)
    if not text.strip():
        return [{"id": "error", "text": "No text could be extracted from this PDF.", "metadata": {"error": "empty_pdf"}}]
    return process_text(text)

def process_docx(file) -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp:
        temp.write(file.getvalue())
        temp_path = temp.name
    doc = docx.Document(temp_path)
    text = "\n\n".join([para.text for para in doc.paragraphs])
    os.unlink(temp_path)
    return process_text(text)

def process_csv(file) -> List[Dict[str, Any]]:
    df = pd.read_csv(file)
    chunks = []
    row_chunk_size = 100
    for i in range(0, len(df), row_chunk_size):
        chunk_df = df.iloc[i:i + row_chunk_size]
        chunk_text = chunk_df.to_string()
        chunk_id = generate_chunk_id(chunk_text)
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {"rows": f"{i}-{min(i + row_chunk_size, len(df))}", "columns": ",".join(df.columns.tolist())}
        })
    return chunks

def process_image(file) -> List[Dict[str, Any]]:
    """Process an image file (PNG/JPG), extracting and correcting text."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
        temp.write(file.getvalue())
        temp_path = temp.name
    try:
        image = cv2.imread(temp_path)
        if image is None:
            raise ValueError(f"Could not read image file at {temp_path}")
        
        # Preprocess images for OCR
        processed_images = [
            image,
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 9, 75, 75),
            cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.5, cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 3), -0.5, 0),
            cv2.adaptiveThreshold(cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ]
        
        reader = easyocr.Reader(['en'])
        all_texts = []
        for img in processed_images:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(img, contrast_ths=0.1, adjust_contrast=0.5, text_threshold=0.5, low_text=0.3)
            if results:
                all_texts.append(" ".join([text[1] for text in results]))
        
        combined_text = " ".join(all_texts)
        if not combined_text.strip():
            # Fallback: Try dilation if no text detected
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), kernel, iterations=1)
            results = reader.readtext(dilated, paragraph=True)
            combined_text = " ".join([text[1] for text in results]) or "No text detected in image"
        
        # Correct spelling before chunking
        corrected_text = correct_spelling(combined_text)
        
        # Process and return chunks
        chunks = process_text(corrected_text)
        for chunk in chunks:
            chunk["metadata"]["type"] = "image_text"
        return chunks
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return [{"id": "error", "text": f"Error processing image: {str(e)}", "metadata": {"error": "image_processing"}}]
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info(f"Cleaned up image file: {temp_path}")

def process_video(file) -> List[Dict[str, Any]]:
    """Process an MP4 or AVI video file, extracting audio transcript first, then corrected in-video text."""
    model = CLIPModel.from_pretrained(MODEL_DIR)
    processor = CLIPProcessor.from_pretrained(MODEL_DIR)
    
    # Save video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
        temp.write(file.getvalue())
        temp_path = temp.name

    try:
        # Open video with OpenCV
        video = cv2.VideoCapture(temp_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video file at {temp_path}")
        
        # Extract video metadata
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        frame_interval = max(1, fps * 2)  # Process every 2 seconds
        
        # Summary chunk for video metadata (stored first)
        video_summary = f"Video with {total_frames} frames, {duration:.2f} seconds duration, {width}x{height} resolution, {fps} FPS."
        chunks = [{
            "id": generate_chunk_id(video_summary),
            "text": video_summary,
            "metadata": {
                "type": "video_summary",
                "duration": duration,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "total_frames": total_frames
            }
        }]

        # Step 1: Extract and transcribe audio (first)
        audio_path = None
        transcript_text = None
        if WHISPER_MODEL:
            try:
                import ffmpeg
                audio_path = os.path.join(tempfile.gettempdir(), f"video_audio_{os.path.basename(temp_path)}.wav")
                logger.info(f"Attempting to extract audio to {audio_path}")
                stream = ffmpeg.input(temp_path)
                stream = ffmpeg.output(stream, audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
                ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
                logger.info(f"Audio extracted to {audio_path}")

                logger.info("Starting Whisper transcription")
                result = WHISPER_MODEL.transcribe(audio_path, task="translate")
                transcript_text = result['text']
                logger.info(f"Audio transcribed: {len(transcript_text)} characters")
            except Exception as e:
                logger.error(f"Error during audio extraction/transcription: {str(e)}")
            finally:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    logger.info(f"Cleaned up audio file: {audio_path}")

        # Process and store audio transcript
        if transcript_text:
            transcript_chunks = process_text(transcript_text, chunk_size=1000, overlap=200)
            for chunk in transcript_chunks:
                chunk["metadata"].update({
                    "type": "video_audio_transcript",
                    "duration": duration,
                    "source": "audio"
                })
            chunks.extend(transcript_chunks)
        else:
            logger.warning("No audio transcript available for this video")

        # Step 2: Extract in-video text using OCR (after audio)
        frame_texts, frame_timestamps = [], []
        frame_count, prev_frame_gray = 0, None
        scene_threshold = 30.0
        reader = easyocr.Reader(['en'])
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            process_this_frame = frame_count % frame_interval == 0
            
            # Detect scene changes
            if prev_frame_gray is not None:
                frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
                if np.mean(frame_diff) > scene_threshold:
                    process_this_frame = True
            prev_frame_gray = current_frame_gray
            
            if process_this_frame:
                timestamp = frame_count / fps if fps > 0 else 0
                frame_timestamps.append(timestamp)
                
                # OCR processing with multiple image enhancements
                processed_frames = [
                    frame,
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.bilateralFilter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 9, 75, 75),
                    cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.5, cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 3), -0.5, 0),
                    cv2.adaptiveThreshold(cv2.bilateralFilter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                ]
                all_texts = []
                for img in processed_frames:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    results = reader.readtext(img, contrast_ths=0.1, adjust_contrast=0.5, text_threshold=0.5, low_text=0.3)
                    if results:
                        all_texts.append(" ".join([text[1] for text in results]))
                
                frame_text = " ".join(all_texts) or f"Video frame at {timestamp:.2f} seconds (no text detected)"
                # Correct spelling for frame text
                corrected_frame_text = correct_spelling(frame_text)
                frame_texts.append(corrected_frame_text)
        
        video.release()
        logger.info(f"Extracted {len(frame_texts)} text segments from video frames")

        # Add frame text chunks
        for i, (text, timestamp) in enumerate(zip(frame_texts, frame_timestamps)):
            enhanced_text = f"At {timestamp:.2f} seconds: {text}"
            chunk_id = generate_chunk_id(enhanced_text)
            chunks.append({
                "id": chunk_id,
                "text": enhanced_text,
                "metadata": {
                    "frame_index": i,
                    "timestamp": timestamp,
                    "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                    "type": "video_frame_text",
                    "has_text": bool(text.strip())
                }
            })

        return chunks

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return [{"id": "error", "text": f"Error processing video: {str(e)}", "metadata": {"error": "video_processing"}}]
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info(f"Cleaned up video file: {temp_path}")

def initialize_index():
    return faiss_db

def save_index():
    logger.info("FAISS index saved to MongoDB")

def process_file(file, file_type: str) -> Dict[str, Any]:
    """Process a file or URL based on its type."""
    if file_type == "youtube":
        chunks = process_youtube(file)
    elif file_type == "pdf":
        chunks = process_pdf(file)
    elif file_type == "docx":
        chunks = process_docx(file)
    elif file_type == "csv":
        chunks = process_csv(file)
    elif file_type == "txt":
        chunks = process_text(file.getvalue().decode('utf-8'))
    elif file_type in ["png", "jpg", "jpeg"]:
        chunks = process_image(file)
    elif file_type in ["mp4", "avi"]:
        chunks = process_video(file)
    else:
        return {"error": f"Unsupported file type: {file_type}"}
    
    if not chunks or ("id" in chunks[0] and chunks[0]["id"] == "error"):
        return {"error": chunks[0]["text"] if chunks else "No chunks generated"}
    
    vectors = [embed_text(chunk["text"]) for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    try:
        faiss_db.add_vectors(vectors, texts, metadatas)
        return {"status": "success", "chunks_added": len(chunks)}
    except Exception as e:
        logger.error(f"Error adding to index: {str(e)}")
        return {"status": "error", "message": str(e)}

def search_documents(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    query_embedding = embed_text(query)
    results = faiss_db.search(query_embedding, k=n_results)
    contexts = []
    for result in results:
        doc = result["text"].replace("**", "").replace("\n", "<br>")
        contexts.append({"text": f"<b>{doc}</b>", "metadata": result["metadata"]})
    return contexts

def get_collection_stats() -> Dict[str, Any]:
    return {
        "total_documents": faiss_db.index.ntotal,
        "vector_count": faiss_db.index.ntotal,
        "storage_type": "FAISS + MongoDB"
    }

def clear_database() -> Dict[str, Any]:
    global faiss_db
    faiss_db = FAISS_VectorDB(dimension)
    try:
        docs_collection.delete_many({})
        logger.info("Cleared MongoDB documents")
    except Exception as e:
        logger.error(f"Error clearing MongoDB: {str(e)}")
    return {"status": "success", "message": "Database cleared"}

def optimize_index() -> Dict[str, Any]:
    return {"status": "info", "message": "Optimization not needed with FAISS IndexFlatL2"}