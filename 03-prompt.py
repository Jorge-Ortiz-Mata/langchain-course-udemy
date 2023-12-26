# Read and ask questions from documents using ChromaDB
# pip install chromadb

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_custom_retriever import ReduntantCustomRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

embeddings=OpenAIEmbeddings()

chat = ChatOpenAI()

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings
)

retriever = ReduntantCustomRetriever(
  embeddings=embeddings,
  chroma=db
)

chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=retriever,
  chain_type="stuff"
)

result = chain.run(
  "What is an interesting fact about the English language?"
)

print(result)