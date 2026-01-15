from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

template = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

user_input = input("Please provide a topic ")

# prompt = template.invoke(user_input)

# result = model.invoke(prompt)

# print(result.content)

# result = chain.invoke(user_input)

chain = template | model | parser

result = chain.invoke({'topic': user_input})

print(result)

chain.get_graph().print_ascii()


 