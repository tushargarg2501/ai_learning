from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b")

response = llm.invoke(
    "In the LangChain framework for building LLM applications, what are guardrails? "
    "Explain practical ways to validate inputs and outputs, constrain tool use, "
    "and enforce structured responses. Do not describe calculus or the chain rule."
)
print(response.content)
