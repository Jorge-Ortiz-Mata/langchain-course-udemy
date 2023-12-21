# Chatbot
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# ConversationBufferMemory is a memory used for models with chat style
# FileChatMessageHistory is used to save the conversation in a file. It works well with ConversationBufferMemory
# ConversationSummaryMemory is used to decrease the amount text sent to the model. It menas less money. It has its own chain

chat = ChatOpenAI(verbose=True)

# The return_messages param stores the strings given by the user and the AI using the HumanPromptTemplate and AIPropmtTemplate
memory = ConversationSummaryMemory(
  memory_key="messages", 
  return_messages=True, 
  llm=chat
  # chat_memory=FileChatMessageHistory("messages_history.json")
)

prompt = ChatPromptTemplate(
  input_variables=["content", "messages"],
  messages=[
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
  content = input(">> ")

  response = chain({ "content": content })

  print(response["text"])
