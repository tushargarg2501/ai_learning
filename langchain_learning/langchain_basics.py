from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b")

response = llm.invoke("what are the guard rails for lang chain?")
print(response.content)
