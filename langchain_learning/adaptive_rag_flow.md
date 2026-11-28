# Adaptive RAG and CRAG Flow

This diagram follows [`adaptive_rag_learning.py`](./adaptive_rag_learning.py). The
router selects one retrieval strategy; CRAG then checks whether the retrieved
evidence is enough before the final answer is generated.

```mermaid
flowchart TD
    A([Start]) --> B[Load coffee HTML pages]
    B --> C[Add category metadata and split into chunks]
    C --> D[Embed with Ollama nomic-embed-text and upsert to coffeeindex-ollama]
    D --> E[/Enter a question/]
    E --> F{Does it name a category<br/>and ask for a guide, page,<br/>category, only, or document?}

    F -- Yes --> G[Self-Query]
    G --> G1[LLM extracts semantic query<br/>and category filter]
    G1 --> J[Retrieve candidate passages]

    F -- No --> H{Does it contain a broad or<br/>comparative cue?}
    H -- Yes --> I[MultiQuery]
    I --> I1[Generate alternate question phrasings<br/>and retrieve for each]
    I1 --> J
    H -- No --> K[Simple similarity search]
    K --> J

    J --> L[Deduplicate candidates]
    L --> M{CRAG: does a local LLM grader<br/>judge the passages sufficient<br/>to answer the question?}
    M -- Yes --> P[Keep local passages]
    M -- No, or no candidates --> N[Try Yahoo, Brave, Startpage, then Mojeek]
    N --> O{Any search results<br/>with snippets?}
    O -- No --> Q([Stop: report no usable evidence])
    O -- Yes --> R[Convert web results to documents]
    P --> S[CrossEncoder reranks evidence]
    R --> S
    S --> T[Keep top 3 passages]
    T --> U[Generate answer using only selected context<br/>and cite filenames or URLs]
    U --> V([Return answer])
```

## What question takes each route?

| Route | When the router chooses it | Example question | What retrieval does |
|---|---|---|---|
| Simple similarity | The question is direct and has no special routing cue | “How is cold brew coffee made?” | Searches Pinecone for semantically similar chunks. |
| MultiQuery | The question is broad or comparative; it includes cues such as “compare”, “difference”, “recommend”, or “options” | “Compare different coffee brewing methods.” | Uses the local LLM to produce alternate phrasings, searches for them, and includes the original question. |
| Self-Query | A known category (`cold_brew`, `espresso`, or `pour_over`) appears alongside a metadata cue such as “guide”, “page”, “only”, or “document” | “Find the cold brew guide.” | Uses the LLM to build a semantic query and a Pinecone metadata filter. |

## CRAG correction scenarios

- **Local evidence is sufficient:** CRAG keeps the retrieved local passages. It
  skips web search, then reranks the passages and answers from the top three.
- **Local evidence is insufficient or retrieval found nothing:** CRAG searches
  Yahoo, Brave, Startpage, then Mojeek in order. It stops at the first provider
  with usable results, converts results with snippets into evidence documents,
  reranks them, and answers from the top three search results.
- **No usable web snippets:** The program stops without generating an answer
  from unsupported context.

The routing rules are keyword-based in this learning example. The CRAG evidence
decision is made by the local Ollama model. When the web fallback runs, the
question is sent to each provider attempted until one returns usable results.
Embeddings use Ollama's `nomic-embed-text` model and the separate Pinecone index
`coffeeindex-ollama`; pull the model with `ollama pull nomic-embed-text` first.
The CrossEncoder reranker still downloads its model from Hugging Face.
