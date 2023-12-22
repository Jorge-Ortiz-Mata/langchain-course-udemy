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

loader = TextLoader("facts.txt")

docs = loader.load_and_split(
  text_splitter=text_splitter
)

db = Chroma.from_documents(
  docs,
  embedding=embeddings,
  persist_directory="emb"
)

results = db.similarity_search_with_score(
    "What is an interesting fact about the English language?"
  )

for result in results:
    print("\n")
    print(result)
    print(result[1])
    print(result[0].page_content)