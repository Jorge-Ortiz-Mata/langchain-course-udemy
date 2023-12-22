from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
# import argparse # Configuration for accepting params from the terminal

load_dotenv()

# parser = argparse.ArgumentParser() # Configuration for accepting params from the terminal
# parser.add_argument("--language", default="ruby") # Configuration for accepting params from the terminal
# parser.add_argument("--task", default="print each value of the next array: ['jorge', 'ortiz', 'mata']") # Configuration for accepting params from the terminal
# args = parser.parse_args() # Configuration for accepting params from the terminal

llm = OpenAI()

# ============== Prompts and Chains ==============

code_generation_prompt = PromptTemplate(
  template="Write a {language} function that will {task}",
  input_variables=["language", "task"]
)

code_validation_prompt = PromptTemplate(
  template="Write a test for the following {language} code using RSpec: \n{code}",
  input_variables=["language", "code"]
)

code_generation_chain = LLMChain(
  llm=llm,
  prompt=code_generation_prompt,
  output_key="code"
)

code_validation_chain = LLMChain(
  llm=llm,
  prompt=code_validation_prompt,
  output_key="test"
)

# ========= Combine chains ========

general_chain = SequentialChain(
  chains=[code_generation_chain, code_validation_chain],
  input_variables=["task", "language"],
  output_variables=["test", "code"]
)

result = general_chain({
  "language": "ruby",
  "task": "return a list of five numbers"
})

print(result["code"])
print(result["test"])


# Question about doing it in a different way

def create_test(question):
  response = llm(question)

  return response

test_one = create_test("Write a test in ruby where a function returns a list of five numbers")
test_two = create_test("Write a test in javascript where a function returns a list of five numbers")

print(test_one)
print(test_two)