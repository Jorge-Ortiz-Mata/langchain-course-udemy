# Read and ask questions from documents using ChromaDB
# pip install chromadb

from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
  separator="\n", # Split text when finding a new line
  chunk_size=300, # Split the text into 300 characters at most.
  chunk_overlap=0 # It takes characters from the previos chunck (100, 200). It is useful on PDF's
)

# Load document file
loader = TextLoader("facts.txt")

# Split document into multiple chuncks
docs = loader.load_and_split(
  text_splitter=text_splitter
)

# Store embedings (documents) in the database
db = Chroma.from_documents(
  docs,
  embedding=embeddings,
  persist_directory="emb"
)

# Find chunks with similar result according to this question.
results = db.similarity_search(
  "What is an interesting fact about the English language?"
)

# Show chunks information
for result in results:
    print("\n")
    print(result.page_content)