from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

template = PromptTemplate(
    template="Generate key 5 facts on {topic}",
    input_variables=['topic']
)

user_input = input("Enter the topic ")

# prompt = template.invoke({'topic': user_input})

# result = model.invoke(prompt)

# print(result.content)

chain = template | model | parser

result = chain.invoke({'topic': user_input})

print(result)

chain.get_graph().print_ascii()

