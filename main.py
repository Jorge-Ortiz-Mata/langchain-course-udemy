from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse # Configuration for accepting params from the terminal

load_dotenv()

parser = argparse.ArgumentParser() # Configuration for accepting params from the terminal
parser.add_argument("--language", default="ruby") # Configuration for accepting params from the terminal
parser.add_argument("--task", default="return an array with names that starts with G") # Configuration for accepting params from the terminal
args = parser.parse_args() # Configuration for accepting params from the terminal

llm = OpenAI()

code_prompt = PromptTemplate(
  template="Write a very short {language} function that will {task}",
  input_variables=["language", "task"]
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt
)

result = code_chain({
  "language": args.language,
  "task": args.task
})

# result = code_chain({
  # "language": "ruby",
  # "task": "print the next 5 numbers starting from 50"
# })

print(result)