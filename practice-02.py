# Chatbot
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

chat = ChatOpenAI()

# The return_messages param stores the strings given by the user and the AI using the HumanPromptTemplate and AIPropmtTemplate
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

prompt = ChatPromptTemplate(
  input_variables=["content", "messages"],
  messages=[
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
  content = input(">> ")

  response = chain({ "content": content })

  print(response["text"])
