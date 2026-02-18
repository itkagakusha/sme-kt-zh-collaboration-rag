# Conversational Toolkit

Conversational Toolkit is a Python library designed to simplify the development of applications using Language Models (LLMs) and agent frameworks. It provides a high-level abstraction for managing conversations, tools, and agents, and is built to integrate smoothly with a dedicated frontend interface: [Conversational Toolkit Frontend](https://gitlab.datascience.ch/industry/common/conversational-toolkit-frontend).

While it can be used independently, its full potential is unlocked when paired with the frontend. If you're working without the frontend, you may want to explore alternatives like LangChain, LlamaIndex, or Guidance, which provide similar abstractions around LLMs.

## Installation
To include this project in your own codebase, add it directly to your `requirements.txt` or install it with pip:

```sh
pip install git+ssh://git@gitlab.datascience.ch/industry/common/conversational-toolkit.git@v6.0.1
```

## Example Usage
Here is a minimal example that shows how to configure and run the toolkit with in-memory components:

```python
from conversational_toolkit.agents.tool_agent import ToolAgent
from conversational_toolkit.api.server import create_app
from conversational_toolkit.conversation_database.controller import ConversationalToolkitController
from conversational_toolkit.conversation_database.in_memory.conversation import InMemoryConversationDatabase
from conversational_toolkit.conversation_database.in_memory.message import InMemoryMessageDatabase
from conversational_toolkit.conversation_database.in_memory.reactions import InMemoryReactionDatabase
from conversational_toolkit.conversation_database.in_memory.source import InMemorySourceDatabase
from conversational_toolkit.conversation_database.in_memory.user import InMemoryUserDatabase
from conversational_toolkit.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from conversational_toolkit.llms.ollama import OllamaLLM
from conversational_toolkit.retriever.vectorstore_retriever import VectorStoreRetriever
from conversational_toolkit.tools.retriever import RetrieverTool
from conversational_toolkit.vectorstores.chromadb import ChromaDBVectorStore

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

conversation_database = InMemoryConversationDatabase("conversations.json")
message_database = InMemoryMessageDatabase("messages.json")
reaction_database = InMemoryReactionDatabase("reactions.json")
source_database = InMemorySourceDatabase("sources.json")
user_database = InMemoryUserDatabase("users.json")
vector_store = ChromaDBVectorStore("chunks.db")

agent = ToolAgent(
    llm=OllamaLLM(
        tools=[
            RetrieverTool(
                name="Retriever",
                description="Retriever is a tool that retrieves information from a database.",
                retriever=VectorStoreRetriever(embedding_model, vector_store, 5),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                llm=OllamaLLM(),
            )
        ]
    ),
    system_prompt="You are a helpful AI assistant specialized in answering question.",
    max_steps=5,
)

controller = ConversationalToolkitController(
    conversation_database,
    message_database,
    reaction_database,
    source_database,
    user_database,
    agent,
)

app = create_app(controller)
```

More examples can be found in the `examples/` folder.

## Contributing
We use [Semantic Versioning](https://semver.org/). To update the version of the project:

- **Major Version:**
```sh
make major
```
- **Minor Version:**
```sh
make minor
```
- **Patch Version:**
```sh
make patch
```

These commands use `bump2version` to update the version and apply it to the `__init__.py` file inside the `conversational_toolkit` package.

---

If you're building agentic applications or conversation managers using LLMs, this toolkit offers a modular and extensible foundation with ready-to-use tools, agent classes, storage layers, and a REST API layer to expose your assistant.
