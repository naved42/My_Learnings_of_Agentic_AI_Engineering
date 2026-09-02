"""YouTube RAG Application using LangChain and HuggingFace embeddings."""
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# Configuration
load_dotenv()

CHROMA_PATH = "./youtube_db"
COLLECTION_NAME = "youtube_rag"
MAX_WORDS = 500
RETRIEVAL_K = 5

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM
llm = ChatGroq(
    model="groq/compound-mini",
    temperature=0.7,
    max_tokens=1000,
)

# Initialize vector store
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)


def get_video_id(url: str) -> str:
    """Extract YouTube video ID from URL or return if already an ID.
    
    Args:
        url: YouTube URL or video ID
        
    Returns:
        11-character video ID
        
    Raises:
        ValueError: If URL format is invalid
    """
    # User entered ID directly
    if len(url) == 11 and "http" not in url:
        return url

    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"shorts/([^?]+)",
        r"embed/([^?]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)[:11]

    raise ValueError("Invalid YouTube URL.")


def get_transcript(video_id: str):
    """Fetch English transcript from YouTube video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        List of transcript snippets
        
    Raises:
        Exception: If transcripts are disabled or not found
    """
    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(video_id, languages=["en"])
        return transcript
    except TranscriptsDisabled as e:
        raise Exception("Captions are disabled for this video.") from e
    except NoTranscriptFound as e:
        raise Exception("No English transcript was found.") from e


def create_documents(transcript, video_id: str):
    """Create Document chunks from transcript.
    
    Args:
        transcript: List of transcript snippets
        video_id: YouTube video ID
        
    Returns:
        List of Document objects with page_content and metadata
    """
    documents = []
    current_text = []
    current_start = None
    current_end = None
    word_count = 0

    for snippet in transcript:
        text = snippet.text.strip()
        if not text:
            continue

        if current_start is None:
            current_start = snippet.start

        current_text.append(text)
        word_count += len(text.split())
        current_end = snippet.start + snippet.duration

        if word_count >= MAX_WORDS:
            documents.append(
                Document(
                    page_content=" ".join(current_text),
                    metadata={
                        "video_id": video_id,
                        "start": current_start,
                        "end": current_end,
                    },
                )
            )
            current_text = []
            word_count = 0
            current_start = None

    # Add remaining text
    if current_text:
        documents.append(
            Document(
                page_content=" ".join(current_text),
                metadata={
                    "video_id": video_id,
                    "start": current_start,
                    "end": current_end,
                },
            )
        )

    return documents


def index_video(url: str) -> str:
    """Index a YouTube video by extracting, chunking, and embedding transcript.
    
    Args:
        url: YouTube URL or video ID
        
    Returns:
        Video ID of indexed video
    """
    print("\nExtracting video ID...")
    video_id = get_video_id(url)
    print(f"Video ID: {video_id}")

    print("Downloading transcript...")
    transcript = get_transcript(video_id)
    print(f"Transcript snippets: {len(transcript)}")

    print("Creating chunks...")
    documents = create_documents(transcript, video_id)
    print(f"Created {len(documents)} chunks.")

    print("Creating embeddings and indexing...")
    vectorstore.add_documents(documents)
    print("\n✓ Video successfully indexed!")

    return video_id


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def ask_question(question: str) -> None:
    """Answer a question using retrieved transcript context.
    
    Args:
        question: User's question about the video
    """
    print("\nSearching transcript...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    documents = retriever.invoke(question)

    if not documents:
        print("No relevant information found.")
        return

    # Build context from retrieved documents
    context_parts = []
    for doc in documents:
        timestamp = format_timestamp(doc.metadata["start"])
        context_parts.append(
            f"[TIMESTAMP: {timestamp}]\n\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Create prompt
    prompt = f"""You are a YouTube video question-answering assistant.

Answer the question using ONLY the transcript context below.

Rules:
1. Do not invent information.
2. If the answer isn't in the transcript, say you couldn't find \
the answer in the video.
3. Be clear and concise.
4. Mention relevant timestamps when possible.

TRANSCRIPT:

{context}

QUESTION:
{question}"""

    print("Generating answer...")
    response = llm.invoke(prompt)

    # Display answer
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response.content)

    # Display sources
    print("\nSOURCES")
    print("-" * 60)

    seen = set()
    for doc in documents:
        video_id = doc.metadata["video_id"]
        start = int(doc.metadata["start"])
        timestamp = format_timestamp(start)
        key = (video_id, start)

        if key in seen:
            continue

        seen.add(key)
        url = f"https://www.youtube.com/watch?v={video_id}&t={start}s"
        print(f"[{timestamp}] {url}")


# ============================================================
# MENU

def main() -> None:
    """Main menu for YouTube RAG application."""
    print(
        "\n╔══════════════════════════════════════╗"
        "\n║          🎥 YouTube RAG              ║"
        "\n╚══════════════════════════════════════╝\n"
    )

    while True:
        print("\n1. Add YouTube video\n2. Ask a question\n3. Exit")
        choice = input("Choice: ").strip()

        if choice == "1":
            url = input("\nYouTube URL: ").strip()
            try:
                index_video(url)
            except Exception as e:
                print(f"\n✗ Error: {e}")

        elif choice == "2":
            question = input("\nQuestion: ").strip()
            if not question:
                print("Please enter a question.")
                continue
            try:
                ask_question(question)
            except Exception as e:
                print(f"\n✗ Error: {e}")

        elif choice == "3":
            print("\nGoodbye! 👋")
            break

        else:
            print("\nInvalid choice.")


if __name__ == "__main__":
    main()

