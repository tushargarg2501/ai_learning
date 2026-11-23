from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str
    hobbies: list[str] = []


parser = PydanticOutputParser(pydantic_object=Person)

# Step 3: Manually format prompt string (no PromptTemplate)
input_text = "Hi, my name is Priya Sharma. I'm twenty and my email is priya.sharma@example.com and hobbies are reading, cooking and singing"
format_instructions = parser.get_format_instructions()

final_prompt = f"""Extract the person's values from this text. Return one JSON object with
the keys name, age, email and hobbies. Do not return the schema or any explanation.

{input_text}
{format_instructions}
"""

llm = ChatOllama(model="llama3.2:3b", format="json")
response = llm.invoke(final_prompt)

try:
    person = parser.parse(response.content)
    print("structured data:", person.model_dump())
    print(person)
except Exception as e:
    print("Error parsing response:", e)
