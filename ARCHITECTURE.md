# Architecture Mapping: Deep Agents vs BeeAI Framework

Questo documento mostra come ogni classe del sistema Deep Agents si ispira alle classi corrispondenti del beeai-framework.

## Architettura Esagonale (Hexagonal Architecture)

**Sì, BeeAI Framework usa l'Architettura Esagonale** (nota anche come Ports & Adapters), e il nostro sistema replica questo pattern.

### Cos'è l'Architettura Esagonale?

L'architettura esagonale, proposta da Alistair Cockburn, separa la logica di business (core) dalle dipendenze esterne (adapters) attraverso interfacce (ports).

```
                    ┌─────────────────────────────────┐
                    │                                 │
                    │         HEXAGON CORE            │
                    │      (Business Logic)           │
                    │                                 │
                    │  ┌──────────────────────────┐   │
                    │  │   Backend (Domain)       │   │
                    │  │  • ChatModel (Port)      │   │
                    │  │  • ChatModelParameters   │   │
                    │  │  • Message, Output       │   │
                    │  └──────────────────────────┘   │
                    │                                 │
                    └────────┬──────────┬─────────────┘
                             │          │
                    ┌────────┴──┐   ┌───┴────────┐
                    │           │   │            │
              ┌─────▼─────┐ ┌──▼───▼──┐ ┌──────▼─────┐
              │  Ollama   │ │ WatsonX │ │  Future    │
              │  Adapter  │ │ Adapter │ │  Adapters  │
              │ (Driver)  │ │(Driver) │ │  (OpenAI,  │
              └───────────┘ └─────────┘ │  Anthropic)│
                                        └────────────┘
```

### Come BeeAI Implementa l'Architettura Esagonale

#### 1. **Core (Hexagon)** - `beeai_framework/backend/`
- **Port (Interface)**: `ChatModel` (classe astratta)
- **Domain Logic**: Gestione messaggi, parametri, factory methods
- **Indipendente** da implementazioni specifiche

#### 2. **Adapters (Outside)** - `beeai_framework/adapters/`
- **Driver Adapters**: Implementazioni specifiche per ogni provider
  - `adapters/ollama/` → `OllamaChatModel`
  - `adapters/watsonx/` → `WatsonxChatModel`
  - `adapters/openai/` → `OpenAIChatModel`
  - `adapters/anthropic/` → `AnthropicChatModel`
  - etc.

#### 3. **Ports (Interfaces)**
```python
# Port = Interfaccia astratta nel core
class ChatModel(ABC):
    @abstractmethod
    async def _run(self, input, options, context):
        pass
```

#### 4. **Dependency Inversion**
- Il **core dipende solo da astrazioni** (ports)
- Gli **adapters dipendono dal core** (implementano i ports)
- **NON** il contrario (core non conosce gli adapters)

### Come il Nostro Sistema Replica l'Architettura Esagonale

#### 1. **Core (Hexagon)** - `src/backend/`
```python
# Port (Interface)
class ChatModel(ABC):
    @abstractmethod
    async def generate(self, messages, stream, **kwargs):
        pass

    @abstractmethod
    async def chat(self, prompt, system_prompt, stream, **kwargs):
        pass
```

#### 2. **Adapters** - `src/adapters/`
```python
# Driver Adapter per Ollama
class OllamaChatModel(ChatModel):
    async def generate(self, messages, stream, **kwargs):
        # Implementazione specifica Ollama
        pass

# Driver Adapter per WatsonX
class WatsonxChatModel(ChatModel):
    async def generate(self, messages, stream, **kwargs):
        # Implementazione specifica WatsonX
        pass
```

#### 3. **Factory Pattern** (Dependency Injection)
```python
# Il core usa factory per creare adapters
class ChatModel:
    @staticmethod
    def from_name(name: str, **kwargs) -> ChatModel:
        config = ChatModelParameters.from_name(name, **kwargs)

        # Dependency Injection: core crea adapter
        if config.provider == "ollama":
            from ..adapters.ollama import OllamaChatModel
            return OllamaChatModel(config)
        elif config.provider == "watsonx":
            from ..adapters.watsonx import WatsonxChatModel
            return WatsonxChatModel(config)
```

### Vantaggi dell'Architettura Esagonale

| Vantaggio          | Descrizione                                   | Esempio nel Nostro Sistema                                    |
| ------------------ | --------------------------------------------- | ------------------------------------------------------------- |
| **Testabilità**    | Core testabile senza dipendenze esterne       | Possiamo testare `ChatModel` con mock adapters                |
| **Sostituibilità** | Adapters intercambiabili                      | Possiamo passare da Ollama a WatsonX senza modificare il core |
| **Indipendenza**   | Core indipendente da tecnologie esterne       | Backend non sa nulla di HTTP, API specifiche                  |
| **Estensibilità**  | Nuovi adapters senza modificare il core       | Aggiungere OpenAI richiede solo nuovo adapter                 |
| **Manutenibilità** | Modifiche agli adapters non impattano il core | Cambiare protocollo Ollama non tocca il backend               |

### Confronto: BeeAI vs Nostro Sistema

| Aspetto          | BeeAI Framework                     | Nostro Sistema                 |
| ---------------- | ----------------------------------- | ------------------------------ |
| **Architettura** | ✅ Esagonale (Ports & Adapters)      | ✅ Esagonale (Ports & Adapters) |
| **Core**         | `backend/` (ChatModel, types)       | `backend/` (ChatModel, types)  |
| **Ports**        | `ChatModel` (abstract)              | `ChatModel` (abstract)         |
| **Adapters**     | `adapters/{provider}/`              | `adapters/{provider}/`         |
| **Factory**      | `ChatModel.from_name()`             | `ChatModel.from_name()`        |
| **DI**           | ✅ Dependency Inversion              | ✅ Dependency Inversion         |
| **Providers**    | 10+ (Ollama, OpenAI, WatsonX, etc.) | 2 (Ollama, WatsonX)            |

### Diagramma Completo: Architettura Esagonale

```
┌─────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                        │
│                      (User Code / Examples)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ uses
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                          HEXAGON CORE                            │
│                    (src/backend/ - Domain)                       │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ChatModel (Port - Abstract Interface)                      │ │
│  │ ─────────────────────────────────────────────────────────  │ │
│  │ + generate(messages, stream, **kwargs)                     │ │
│  │ + chat(prompt, system_prompt, stream, **kwargs)            │ │
│  │ + from_name(name, **kwargs) → ChatModel [Factory]          │ │
│  │ + create(parameters) → ChatModel [Factory]                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ChatModelParameters (Domain Model)                         │ │
│  │ ─────────────────────────────────────────────────────────  │ │
│  │ • provider, protocol, model                                │ │
│  │ • base_url, api_key, project_id                            │ │
│  │ • temperature, max_tokens, stream                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Message, ChatModelOutput (Value Objects)                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
└───────────────────────┬───────────────────┬───────────────────────┘
                        │                   │
                        │ implements        │ implements
                        │                   │
        ┌───────────────▼──────┐    ┌──────▼────────────────┐
        │                      │    │                       │
┌───────▼──────────────────┐   │    │   ┌──────────────────▼──────┐
│  OLLAMA ADAPTER          │   │    │   │  WATSONX ADAPTER        │
│  (src/adapters/ollama/)  │   │    │   │  (src/adapters/watsonx/)│
│                          │   │    │   │                         │
│  OllamaChatModel         │   │    │   │  WatsonxChatModel       │
│  implements ChatModel    │   │    │   │  implements ChatModel   │
│                          │   │    │   │                         │
│  • Ollama Protocol       │   │    │   │  • WatsonX API          │
│  • OpenAI Protocol       │   │    │   │  • IBM Authentication   │
│  • HTTP Client (httpx)   │   │    │   │  • HTTP Client (httpx)  │
└──────────────────────────┘   │    │   └─────────────────────────┘
                               │    │
                               │    │
                    ┌──────────▼────▼──────────┐
                    │   FUTURE ADAPTERS        │
                    │   • OpenAI               │
                    │   • Anthropic            │
                    │   • Google Gemini        │
                    │   • Azure OpenAI         │
                    └──────────────────────────┘
```

### Principi SOLID nell'Architettura Esagonale

1. **Single Responsibility**: Ogni adapter gestisce un solo provider
2. **Open/Closed**: Core chiuso, estensibile con nuovi adapters
3. **Liskov Substitution**: Ogni adapter può sostituire ChatModel
4. **Interface Segregation**: Port minimale (generate, chat)
5. **Dependency Inversion**: Core dipende da astrazioni, non da implementazioni


## Nomenclatura Allineata a BeeAI

Per mantenere coerenza con BeeAI Framework, usiamo gli stessi nomi:

| Nostro Nome Originale | Nome BeeAI              | Descrizione                 |
| --------------------- | ----------------------- | --------------------------- |
| `BaseLLMClient`       | `ChatModel`             | Classe base astratta        |
| `LLMConfig`           | `ChatModelParameters`   | Parametri di configurazione |
| `LLMResponse`         | `ChatModelOutput`       | Output del modello          |
| `OllamaClient`        | `OllamaChatModel`       | Implementazione Ollama      |
| `WatsonXClient`       | `WatsonxChatModel`      | Implementazione WatsonX     |
| `LLMFactory`          | `ChatModel.from_name()` | Factory method              |

**Nota**: Manteniamo la struttura semplificata attuale (`src/llm_proxy/`) ma con nomenclatura allineata a BeeAI.

## Mappatura delle Classi

### 1. **LLMConfig** → `ChatModelParameters` + Provider Name Pattern

**Nostro codice:** `src/llm_proxy/config.py`
```python
class LLMConfig(BaseModel):
    provider: ProviderType
    protocol: ProtocolType
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    temperature: float
    max_tokens: Optional[int]
    stream: bool
```

**BeeAI Framework:** `beeai_framework/backend/types.py` + pattern `provider:model`
```python
class ChatModelParameters(BaseModel):
    max_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    # ...
```

**Ispirazione:**
- BeeAI usa il pattern `"provider:model"` (es. `"ollama:llama3.1"`, `"watsonx:granite-3-8b"`)
- Noi abbiamo creato `LLMConfig.from_name()` che replica questo pattern
- Abbiamo aggiunto `ProviderType` e `ProtocolType` per gestire provider e protocolli multipli

---

### 2. **BaseLLMClient** → `ChatModel` (Abstract Base Class)

**Nostro codice:** `src/llm_proxy/base.py`
```python
class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], ...) -> Union[LLMResponse, AsyncIterator[str]]:
        pass

    @abstractmethod
    async def chat(self, prompt: str, ...) -> Union[LLMResponse, AsyncIterator[str]]:
        pass
```

**BeeAI Framework:** `beeai_framework/backend/chat.py`
```python
class ChatModel(Runnable[ChatModelOutput]):
    @abstractmethod
    async def _run(
        self,
        input: ChatModelInput,
        options: ChatModelOptions,
        run_context: RunContext[ChatModelOutput],
    ) -> ChatModelOutput:
        pass
```

**Ispirazione:**
- BeeAI ha `ChatModel` come classe base astratta per tutti i modelli
- Noi abbiamo `BaseLLMClient` con metodi `generate()` e `chat()`
- Entrambi supportano async/await e context manager (`__aenter__`, `__aexit__`)

---

### 3. **LLMFactory** → `ChatModel.from_name()` + `Backend.from_provider()`

**Nostro codice:** `src/llm_proxy/factory.py`
```python
class LLMFactory:
    @staticmethod
    def from_name(name: str, protocol: Optional[str] = None, **kwargs) -> BaseLLMClient:
        config = LLMConfig.from_name(name, **kwargs)
        return LLMFactory.create(config)

    @staticmethod
    def from_env(...) -> BaseLLMClient:
        # Crea client da variabili d'ambiente
```

**BeeAI Framework:** `beeai_framework/backend/chat.py` + `backend.py`
```python
class ChatModel:
    @staticmethod
    def from_name(name: str | ProviderName, ...) -> "ChatModel":
        # Factory method per creare modelli da nome

class Backend:
    @staticmethod
    def from_provider(name: str | ProviderName) -> "Backend":
        # Crea backend da provider name
```

**Ispirazione:**
- BeeAI usa `ChatModel.from_name("ollama:llama3.1")` per creare istanze
- BeeAI usa `Backend.from_provider("ollama")` per creare backend completi
- Noi abbiamo `LLMFactory.from_name()` e `LLMFactory.from_env()` per flessibilità

---

### 4. **OllamaClient** → `OllamaChatModel` (LiteLLM-based)

**Nostro codice:** `src/llm_proxy/providers/ollama.py`
```python
class OllamaClient(BaseLLMClient):
    async def _generate_ollama(self, messages, stream, **kwargs):
        # Implementazione protocollo Ollama nativo

    async def _generate_openai(self, messages, stream, **kwargs):
        # Implementazione protocollo OpenAI-compatible
```

**BeeAI Framework:** `beeai_framework/adapters/ollama/backend/chat.py`
```python
class OllamaChatModel(LiteLLMChatModel):
    # Usa LiteLLM per gestire Ollama
```

**Ispirazione:**
- BeeAI usa `OllamaChatModel` che estende `LiteLLMChatModel`
- Noi abbiamo implementato direttamente il supporto per due protocolli:
  - Protocollo Ollama nativo (`/api/chat`)
  - Protocollo OpenAI-compatible (`/v1/chat/completions`)
- Questo permette maggiore flessibilità nella scelta del protocollo

---

### 5. **WatsonXClient** → `WatsonxChatModel`

**Nostro codice:** `src/llm_proxy/providers/watsonx.py`
```python
class WatsonXClient(BaseLLMClient):
    async def generate(self, messages, stream, **kwargs):
        # Implementazione WatsonX API
        url = f"{self.base_url}/ml/v1/text/generation"
```

**BeeAI Framework:** `beeai_framework/adapters/watsonx/backend/chat.py`
```python
class WatsonxChatModel(LiteLLMChatModel):
    # Usa LiteLLM per gestire WatsonX
```

**Ispirazione:**
- BeeAI usa `WatsonxChatModel` con LiteLLM
- Noi abbiamo implementato direttamente l'API WatsonX
- Supporto per autenticazione con API key e project ID
- Gestione di streaming e non-streaming

---

### 6. **Message** e **LLMResponse** → `UserMessage`, `AssistantMessage`, `ChatModelOutput`

**Nostro codice:** `src/llm_proxy/base.py`
```python
class Message(BaseModel):
    role: str
    content: str

class LLMResponse(BaseModel):
    content: str
    model: str
    finish_reason: Optional[str]
    usage: Optional[dict]
```

**BeeAI Framework:** `beeai_framework/backend/message.py` + `types.py`
```python
class UserMessage(BaseMessage):
    role: Literal["user"] = "user"
    content: MessageContent

class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    content: MessageContent

class ChatModelOutput(RunnableOutput):
    usage: ChatModelUsage
    output: list[BaseMessage]
```

**Ispirazione:**
- BeeAI ha una gerarchia complessa di messaggi (`UserMessage`, `AssistantMessage`, `ToolMessage`)
- Noi abbiamo semplificato con una singola classe `Message` con campo `role`
- `LLMResponse` è ispirato a `ChatModelOutput` con informazioni su usage e finish_reason

---

## Pattern Chiave Adottati

### 1. **Provider Name Pattern**
```python
# BeeAI
model = ChatModel.from_name("ollama:llama3.1")
model = ChatModel.from_name("watsonx:granite-3-8b")

# Nostro
client = LLMFactory.from_name("ollama:llama3.1")
client = LLMFactory.from_name("watsonx:granite-3-8b")
```

### 2. **Factory Pattern**
```python
# BeeAI
backend = Backend.from_provider("ollama")
model = ChatModel.from_name("openai:gpt-4")

# Nostro
client = LLMFactory.from_name("ollama:llama3.1")
client = LLMFactory.from_env()
```

### 3. **Async/Await + Context Manager**
```python
# BeeAI
async with ChatModel.from_name("ollama:llama3.1") as model:
    response = await model.run([UserMessage("Hello")])

# Nostro
async with LLMFactory.from_name("ollama:llama3.1") as client:
    response = await client.chat("Hello")
```

### 4. **Streaming Support**
```python
# BeeAI
response = await model.run([UserMessage("Hello")], stream=True)

# Nostro
response = await client.chat("Hello", stream=True)
async for chunk in response:
    print(chunk)
```

---

## Differenze Principali

### 1. **Semplificazione**
- **BeeAI**: Sistema completo con agenti, tools, memory, middleware
- **Nostro**: Focus solo sul proxy LLM con configurazione flessibile

### 2. **Protocolli Multipli**
- **BeeAI**: Usa principalmente LiteLLM per gestire provider diversi
- **Nostro**: Supporto esplicito per protocolli multipli (Ollama può usare protocollo nativo o OpenAI)

### 3. **Configurazione**
- **BeeAI**: Configurazione tramite environment variables e parametri
- **Nostro**: `LLMConfig` centralizzato + `from_name()` + `from_env()`

### 4. **Provider Supportati**
- **BeeAI**: Ollama, OpenAI, WatsonX, Anthropic, Groq, Gemini, Azure, Bedrock, etc.
- **Nostro**: Ollama (con 2 protocolli) e WatsonX (focus su semplicità)

---

## Conclusione

Il sistema LLM Proxy è una **versione semplificata e focalizzata** del backend di beeai-framework, che:

1. ✅ Adotta il pattern `provider:model` per la creazione di client
2. ✅ Usa factory methods (`from_name`, `from_env`) per flessibilità
3. ✅ Supporta async/await e context manager
4. ✅ Gestisce streaming e non-streaming
5. ✅ Permette protocolli multipli per lo stesso provider (Ollama)
6. ✅ Mantiene una struttura semplice e comprensibile

## Gestione della Configurazione in BeeAI

### Come BeeAI Gestisce i Parametri

BeeAI **NON usa file YAML** per la configurazione. Usa un sistema a **3 livelli di priorità**:

#### 1. **Parametri Espliciti** (Priorità Massima)
```python
# Passati direttamente al costruttore
model = WatsonxChatModel(
    "granite-3-8b",
    settings={
        "project_id": "my-project",
        "api_key": "my-key",
        "base_url": "https://watsonx.example.com"
    }
)
```

#### 2. **Variabili d'Ambiente** (Priorità Media)
```python
# BeeAI cerca automaticamente variabili d'ambiente specifiche
# Per WatsonX:
WATSONX_PROJECT_ID=my-project
WATSONX_API_KEY=my-key
WATSONX_URL=https://watsonx.example.com
WATSONX_CHAT_MODEL=ibm/granite-3-8b

# Per Ollama:
OLLAMA_CHAT_MODEL=llama3.1
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_API_KEY=ollama
```

#### 3. **Valori di Default** (Priorità Minima)
```python
# Se nessun parametro è fornito, usa i default
model = OllamaChatModel()  # usa "llama3.1" e "http://localhost:11434"
```

### Metodo `_assert_setting_value()`

BeeAI usa un metodo interno per gestire questa priorità:

```python
def _assert_setting_value(
    self,
    name: str,
    value: Any,
    envs: list[str],
    fallback: str | None = None,
    aliases: list[str] | None = None,
    allow_empty: bool = False
):
    # 1. Cerca nel parametro esplicito
    value = value or self._settings.get(name)

    # 2. Se non trovato, cerca negli alias e env vars
    if not value:
        value = next(
            chain(
                (self._settings[alias] for alias in aliases if self._settings.get(alias)),
                (os.environ[env] for env in envs if os.environ.get(env)),
            ),
            fallback,  # 3. Usa fallback se nessuno trovato
        )

    # 4. Solleva errore se richiesto e non trovato
    if not value and not allow_empty:
        raise ValueError(f"Setting {name} is required...")

    self._settings[name] = value
```

### Esempio Pratico: WatsonX

```python
# In beeai_framework/adapters/watsonx/backend/chat.py

class WatsonxChatModel(LiteLLMChatModel):
    def __init__(self, model_id, project_id=None, api_key=None, base_url=None, **kwargs):
        super().__init__(
            model_id if model_id else os.getenv("WATSONX_CHAT_MODEL", "ibm/granite-3-3-8b-instruct"),
            provider_id="watsonx",
            **kwargs,
        )

        # Cerca project_id in: parametro → env var → errore
        self._assert_setting_value("project_id", project_id, envs=["WATSONX_PROJECT_ID"])

        # Cerca base_url in: parametro → env var → fallback calcolato
        self._assert_setting_value(
            "base_url",
            base_url,
            envs=["WATSONX_URL"],
            fallback=f"https://{self._settings['region']}.ml.cloud.ibm.com"
        )
```

### Esempio Pratico: Ollama

```python
# In beeai_framework/adapters/ollama/backend/chat.py

class OllamaChatModel(LiteLLMChatModel):
    def __init__(self, model_id=None, api_key=None, base_url=None, **kwargs):
        super().__init__(
            model_id if model_id else os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
            provider_id="openai",
            **kwargs,
        )

        # api_key con fallback "ollama"
        self._assert_setting_value("api_key", api_key, envs=["OLLAMA_API_KEY"], fallback="ollama")

        # base_url con fallback localhost
        self._assert_setting_value(
            "base_url",
            base_url,
            envs=["OLLAMA_API_BASE"],
            fallback="http://localhost:11434"
        )
```

### File `.env` con `python-dotenv`

BeeAI usa `python-dotenv` per caricare variabili da file `.env`:

```python
from dotenv import load_dotenv

load_dotenv()  # Carica .env nella sessione corrente
```

Esempio `.env`:
```bash
# WatsonX
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_API_KEY=your-api-key
WATSONX_PROJECT_ID=your-project-id
WATSONX_CHAT_MODEL=ibm/granite-3-8b-instruct

# Ollama
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1
```

### Come Abbiamo Replicato Questo

Nel nostro sistema:

1. **`LLMConfig`** supporta parametri espliciti
2. **`LLMFactory.from_env()`** legge da variabili d'ambiente
3. **Valori di default** in `LLMConfig.from_name()`

```python
# Nostro approccio
client = LLMFactory.from_name(
    "watsonx:granite-3-8b",
    base_url="https://...",  # Parametro esplicito
    api_key="..."
)

# O da environment
os.environ["LLM_PROVIDER"] = "watsonx"
os.environ["LLM_MODEL"] = "granite-3-8b"
os.environ["LLM_BASE_URL"] = "https://..."
client = LLMFactory.from_env()
```

### Differenze Chiave

| Aspetto         | BeeAI                                             | Nostro Sistema            |
| --------------- | ------------------------------------------------- | ------------------------- |
| **File Config** | ❌ No YAML, solo `.env`                            | ❌ No YAML, solo `.env`    |
| **Priorità**    | Parametri → Env → Default                         | Parametri → Env → Default |
| **Env Vars**    | Specifiche per provider (`WATSONX_*`, `OLLAMA_*`) | Generiche (`LLM_*`)       |
| **Validazione** | `_assert_setting_value()` con errori              | Pydantic validation       |
| **Fallback**    | Supporto per valori di default dinamici           | Valori statici in config  |

---

**Riferimenti BeeAI Framework:**
- `beeai_framework/backend/chat.py` → `ChatModel` base class
- `beeai_framework/backend/backend.py` → `Backend.from_provider()`
- `beeai_framework/adapters/ollama/` → Implementazione Ollama
- `beeai_framework/adapters/watsonx/` → Implementazione WatsonX
- `beeai_framework/adapters/litellm/chat.py` → `_assert_setting_value()` method

## Schema delle Dipendenze: Backend ↔ Adapters

### Struttura delle Dipendenze

```
┌─────────────────────────────────────────────────────────────┐
│                        Backend Layer                         │
│                     (src/backend/)                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ types.py                                             │   │
│  │ ─────────────────────────────────────────────────── │   │
│  │ • ChatModelParameters (BaseModel)                    │   │
│  │   - provider: ProviderType                           │   │
│  │   - protocol: ProtocolType                           │   │
│  │   - model: str                                       │   │
│  │   - base_url, api_key, project_id                    │   │
│  │   - temperature, max_tokens, stream                  │   │
│  │   + from_name(name: str) → ChatModelParameters       │   │
│  │                                                       │   │
│  │ • ProviderType (Enum): OLLAMA, WATSONX               │   │
│  │ • ProtocolType (Enum): OLLAMA, OPENAI, WATSONX       │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ▲                                  │
│                           │ imports                          │
│                           │                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ chat.py                                              │   │
│  │ ─────────────────────────────────────────────────── │   │
│  │ • Message (BaseModel)                                │   │
│  │   - role: str                                        │   │
│  │   - content: str                                     │   │
│  │                                                       │   │
│  │ • ChatModelOutput (BaseModel)                        │   │
│  │   - content: str                                     │   │
│  │   - model: str                                       │   │
│  │   - finish_reason: Optional[str]                     │   │
│  │   - usage: Optional[dict]                            │   │
│  │                                                       │   │
│  │ • ChatModel (ABC)                                    │   │
│  │   + __init__(config: ChatModelParameters)            │   │
│  │   + generate(messages, stream, **kwargs)             │   │
│  │   + chat(prompt, system_prompt, stream, **kwargs)    │   │
│  │   + create(parameters) → ChatModel [Factory]         │   │
│  │   + from_name(name, protocol, **kwargs) → ChatModel  │   │
│  │   + from_env(provider_env, model_env) → ChatModel    │   │
│  │   + close(), __aenter__(), __aexit__()               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            │ extends & imports
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Ollama Adapter      │              │  WatsonX Adapter     │
│  (src/adapters/      │              │  (src/adapters/      │
│   ollama/)           │              │   watsonx/)          │
├──────────────────────┤              ├──────────────────────┤
│                      │              │                      │
│ OllamaChatModel      │              │ WatsonxChatModel     │
│ extends ChatModel    │              │ extends ChatModel    │
│                      │              │                      │
│ Dependencies:        │              │ Dependencies:        │
│ • ChatModel          │              │ • ChatModel          │
│ • ChatModelOutput    │              │ • ChatModelOutput    │
│ • Message            │              │ • Message            │
│ • ChatModelParameters│              │ • ChatModelParameters│
│ • ProtocolType       │              │                      │
│                      │              │                      │
│ Implements:          │              │ Implements:          │
│ • generate()         │              │ • generate()         │
│ • chat()             │              │ • chat()             │
│ • close()            │              │ • close()            │
│ • __aenter__()       │              │ • __aenter__()       │
│ • __aexit__()        │              │ • __aexit__()        │
│                      │              │                      │
│ Protocol Support:    │              │ Protocol Support:    │
│ • Ollama native      │              │ • WatsonX API        │
│ • OpenAI compatible  │              │                      │
└──────────────────────┘              └──────────────────────┘
```

### Flusso delle Dipendenze

#### 1. **Backend → Adapters** (Dependency Injection)

```python
# Backend definisce l'interfaccia astratta
class ChatModel(ABC):
    def __init__(self, config: ChatModelParameters):
        self.config = config

    @abstractmethod
    async def generate(self, messages, stream, **kwargs):
        pass

# Adapters implementano l'interfaccia
class OllamaChatModel(ChatModel):
    # Implementazione specifica per Ollama
    pass

class WatsonxChatModel(ChatModel):
    # Implementazione specifica per WatsonX
    pass
```

#### 2. **Factory Pattern** (Backend crea Adapters)

```python
# In backend/chat.py
class ChatModel:
    @staticmethod
    def create(parameters: ChatModelParameters) -> ChatModel:
        if parameters.provider == "ollama":
            from ..adapters.ollama import OllamaChatModel
            return OllamaChatModel(parameters)
        elif parameters.provider == "watsonx":
            from ..adapters.watsonx import WatsonxChatModel
            return WatsonxChatModel(parameters)

    @staticmethod
    def from_name(name: str, **kwargs) -> ChatModel:
        config = ChatModelParameters.from_name(name, **kwargs)
        # Usa create() per istanziare l'adapter corretto
        return ChatModel.create(config)
```

#### 3. **Import Dependencies**

```python
# src/adapters/ollama/chat.py
from ...backend.chat import ChatModel, ChatModelOutput, Message
from ...backend.types import ChatModelParameters, ProtocolType

class OllamaChatModel(ChatModel):
    def __init__(self, config: ChatModelParameters):
        super().__init__(config)
        # Usa ProtocolType per determinare il protocollo
        if self.config.protocol == ProtocolType.OPENAI:
            # Usa protocollo OpenAI
        else:
            # Usa protocollo Ollama nativo
```

```python
# src/adapters/watsonx/chat.py
from ...backend.chat import ChatModel, ChatModelOutput, Message
from ...backend.types import ChatModelParameters

class WatsonxChatModel(ChatModel):
    def __init__(self, config: ChatModelParameters):
        super().__init__(config)
        # Usa config per configurare il client WatsonX
```

### Principi Architetturali

#### 1. **Dependency Inversion Principle (DIP)**
- **Backend** definisce interfacce astratte (`ChatModel`, `Message`, `ChatModelOutput`)
- **Adapters** dipendono dalle astrazioni del backend, non viceversa
- Il backend non conosce i dettagli implementativi degli adapters

#### 2. **Open/Closed Principle (OCP)**
- Il backend è **chiuso per modifiche** (interfaccia stabile)
- Il sistema è **aperto per estensioni** (nuovi adapters possono essere aggiunti)
- Aggiungere un nuovo provider richiede solo:
  1. Creare nuovo adapter che estende `ChatModel`
  2. Aggiungere provider a `ProviderType` enum
  3. Aggiungere case nel factory method

#### 3. **Single Responsibility Principle (SRP)**
- **Backend/types.py**: Gestisce configurazione e parametri
- **Backend/chat.py**: Definisce interfaccia e factory methods
- **Adapters**: Implementano protocolli specifici dei provider

#### 4. **Liskov Substitution Principle (LSP)**
- Ogni adapter può sostituire `ChatModel` senza rompere il codice client
- Tutti gli adapters rispettano il contratto definito da `ChatModel`

### Vantaggi di Questa Architettura

1. ✅ **Testabilità**: Backend e adapters possono essere testati indipendentemente
2. ✅ **Manutenibilità**: Modifiche agli adapters non impattano il backend
3. ✅ **Estensibilità**: Nuovi provider possono essere aggiunti facilmente
4. ✅ **Riusabilità**: Il backend può essere riutilizzato con diversi adapters
5. ✅ **Separazione delle Responsabilità**: Backend gestisce logica comune, adapters gestiscono dettagli specifici

### Esempio di Utilizzo

```python
# L'utente interagisce solo con il backend
from src.backend.chat import ChatModel

# Il backend crea automaticamente l'adapter corretto
async with ChatModel.from_name("ollama:llama3.1") as model:
    response = await model.chat("Hello!")
    print(response.content)

# Oppure con WatsonX
async with ChatModel.from_name("watsonx:granite-3-8b",
                                base_url="...",
                                api_key="...") as model:
    response = await model.chat("Hello!")
    print(response.content)
```

L'utente non ha bisogno di conoscere i dettagli degli adapters - il backend gestisce tutto tramite il factory pattern.
