from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a deatiled report on the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following report \n {text}",
    input_variables=['text']
)

user_input = input("Enter the topic ")

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': user_input})

print(result)

chain.get_graph().print_ascii()
