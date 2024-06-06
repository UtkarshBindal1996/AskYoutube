import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from deepmultilingualpunctuation import PunctuationModel
import youtube_transcript_api
from validators import url  # Install validators library (pip install validators)
import urllib
import google.generativeai as genai

generation_config = {
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

genai.configure(api_key=st.secrets["google_secret"])

@st.cache_data
def get_youtube_video_id(url):
  """Extracts the video ID from a YouTube URL.

  Args:
    url: The YouTube URL.

  Returns:
    The video ID, or None if the URL is not a valid YouTube watch URL.
  """

  # Check if the URL is a valid YouTube watch URL.
  if not ("youtube.com/watch" in url or "youtu.be" in url):
    return None

  # Extract the video ID from the URL.
  u = urllib.parse.urlparse(url)
  if "v=" in u.query:
    # www.youtube.com/watch?v=...
    video_id = u.query.split("=")[1]
  elif u.path.split("/")[-1]:
    # youtu.be/....
    video_id = u.path.split("/")[-1]
  else:
    return None
  return video_id

@st.cache_data
def load_data():
  model = SentenceTransformer("all-MiniLM-L6-v2")
  punctModel = PunctuationModel()
  llm = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
  )
  return model, punctModel, llm

model, puctModel, llm = load_data()

@st.cache_data
def embed_document(document_chunks):
  if document_chunks != None:
    # Split document into chunks and generate embeddings (replace with your chunking logic)
    embeddings = model.encode(document_chunks)
    return embeddings
  return None

@st.cache_data
def answer_question(question):
  question_embedding = model.encode([question])
  similarities = model.similarity(question_embedding, st.session_state['document_embeddings'])

  n = 20
  if len(similarities) >= 20:
    top_n_indices = torch.topk(similarities, k=n, dim=1)[1].squeeze(0)
  else:
    top_n_indices = list(range(len(similarities)))

  context = ""
  for indice in top_n_indices:
    context = context + st.session_state['document_chunks'][indice]

  # Get response from LLM
  chat = llm.start_chat()
  response = chat.send_message("I want you to answer the question based on the context in brief. The context is the following: " + context + ". And the question is the following: " + question)  
  return response

@st.cache_data
def process_video(youtube_url):
    try:
        is_valid_url = url(youtube_url)
        if is_valid_url:
          st.write(youtube_url)
          # Extract transcript using youtube_transcript_api
          document_chunks = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(get_youtube_video_id(youtube_url))
          document_chunks = [item['text'] for item in document_chunks]
          document_chunks = " ".join(document_chunks)

          document_chunks = puctModel.restore_punctuation(document_chunks)

          document_chunks = document_chunks.split(".")
          document_embeddings = embed_document(document_chunks)
          return document_embeddings, document_chunks
        else:
          return None, None
    except Exception as e:
        # Handle potential errors (e.g., invalid URL, API quota limitations)
        st.write(f"Error fetching transcript: {e}")
        return None

if "document_embeddings" not in st.session_state:
  st.session_state['document_embeddings'] = None
if "document_chunks" not in st.session_state:
  st.session_state['document_chunks'] = None
if "question_enabled" not in st.session_state:
  st.session_state['question_enabled'] = False
if "url" not in st.session_state:
  st.session_state['url'] = ""

st.title("Ask Youtube")
st.session_state['url'] = st.text_input("Enter YouTube URL (choose a smaller video (<5mins) to avoid memory issues):")
is_valid_url = url(st.session_state['url'])
if not is_valid_url:
   st.write("The entered URL is not valid.")

# Button to fetch transcript
if st.button("Fetch Transcript", disabled=not is_valid_url):
    st.session_state['document_embeddings'], st.session_state['document_chunks'] = process_video(st.session_state['url'])
    st.session_state['question_enabled'] = True

# Enable/disable question input field based on transcript availability
question_input = st.text_input("Ask a question (disabled till transcript is processed):", disabled=not st.session_state['question_enabled'], placeholder="Please enter your question and press the Get Asnwer button below")

# Button to submit question and get answer
if st.session_state['document_embeddings'] is not None and question_input:
    if st.button("Get Answer"):  # Button to trigger answer retrieval
        response = answer_question(question_input)
        st.write(response.text)
