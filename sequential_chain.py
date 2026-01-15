from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt_1 = PromptTemplate(
    template="Generate a detailed report on the {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following report \n {text}",
    input_variables=['text']
)

user_input = input("Please provide a topic ")

chain = prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({'topic': user_input})

print(result)

chain.get_graph().print_ascii()