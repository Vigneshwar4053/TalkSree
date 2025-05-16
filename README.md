
# 🧠 Sree AI Assistant

A conversational AI assistant featuring long-term memory, multilingual dialogue (English + Telugu),helps in coding DSA and customizable personality traits. Powered by local LLM inference via `llama.cpp` and built with Python and Tkinter for a responsive, themed desktop UI.

---

## 🔑 Key Features

### 💬 Conversational Intelligence
- **Short-term Memory:** Tracks the last 10 user-assistant exchanges
- **Long-term Memory:** JSON-based persistent memory with importance-based retention
- **Memory Decay:** Fades irrelevant data over time to keep context efficient
- **Context Awareness:** Dynamic prompt construction for more natural dialogue

### 🧠 Local Language Model Inference
- **Powered by Llama.cpp:** Uses local `.gguf` models
- **Multithreaded Generation:** Keeps UI responsive while generating answers
- **Retry Mechanism:** Automatically retries failed generations

### 🌐 Multilingual Dialogue
- **Primary Language:** English
- **Telugu Phrases:** Integrated support for bilingual conversations
- **Code-Switching:** Seamless transition between languages based on context

### 🎨 Graphical Interface
- **Built with Tkinter:** Lightweight desktop interface
- **Real-time Typing Indicators**
- **Theme Support:** Light, Dark, and Custom themes
- **Conversation History Display**

---

## 🖥️ System Requirements

- **Python** 3.8+
- **RAM:** Minimum 8GB (16GB recommended for optimal LLM performance)
- **LLM Model:** `.gguf` format (e.g., `Meta-Llama-3-8B-Instruct.Q4_0.gguf`)

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/sree-assistant.git
cd sree-assistant
````

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download LLM Model

* Get a `.gguf` model from a trusted provider (e.g., HuggingFace)
* Place it in the project root
* Update `MODEL_PATH` in `sree_assistant.py` if necessary

---

## ⚙️ Configuration

### 🔧 Environment Variables

Create a `.env` file in the root folder:

```env
# Example Configuration
MEMORY_FILE=sree_memory.json
MAX_MEMORY_ITEMS=1000
DEFAULT_THEME=dark
MODEL_PATH=Meta-Llama-3-8B-Instruct.Q4_0.gguf
```

### 👤 Personality Customization

Adjust these fields in the `SreeAssistant` class to change tone and behavior:

* `personality_traits`
* `telugu_phrases`
* `response_styles`
* `capabilities`

---

## 🚀 Usage

### 1. Run the Application

```bash
python sree_assistant.py
```

### 2. Interface Controls

* Type your message in the input field
* Press **Enter** or click **Send**
* Use **Shift+Enter** for multiline input
* Change theme via dropdown menu

### 3. Key Functionalities

* Context-aware, multilingual responses
* Retains relevant long-term memory across sessions
* Configurable assistant behavior and personality

---

## 🧱 Architecture Overview

```
Sree AI Assistant
├── 🧠 Memory Manager
│   ├── Short-term Memory
│   ├── Long-term Memory (JSON)
│   ├── Preference Extraction
│   └── Memory Decay & Optimization
│
├── 🤖 Response Generator
│   ├── Local LLM Interface (llama.cpp)
│   ├── Prompt Construction
│   ├── Threaded Response Queue
│   └── Post-Processing Filters
│
└── 🖼️ Graphical Interface (Tkinter)
    ├── Chat UI Renderer
    ├── Typing Animation
    ├── Theme & Style Manager
    └── Input Control
```

---

## 📚 API Documentation

### `MemoryManager` Class

| Method                                                    | Description                       |
| --------------------------------------------------------- | --------------------------------- |
| `add_conversation_memory(user_input, assistant_response)` | Stores interaction in memory      |
| `get_relevant_memories(query, limit=5)`                   | Retrieves related memories        |
| `save_long_term_memory()`                                 | Persists long-term memory to disk |
| `format_memory_for_context(query)`                        | Formats memory block for prompt   |

### `ResponseGenerator` Class

| Method                             | Description                  |
| ---------------------------------- | ---------------------------- |
| `queue_response(prompt, callback)` | Starts threaded response     |
| `_generate_llm_response(prompt)`   | Generates response using LLM |

### `SreeAssistant` Class

| Method                                    | Description                            |
| ----------------------------------------- | -------------------------------------- |
| `generate_response(user_input, callback)` | Central generation method              |
| `_build_prompt(user_input)`               | Assembles prompt from memory and input |
| `_process_response(response)`             | Refines raw LLM output                 |

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. **Fork** the repository
2. **Create a branch** (`feature/your-feature`)
3. **Commit** your changes
4. **Push** to your branch
5. **Submit a Pull Request**

Please include tests where applicable.

---

## 📫 Support

Have a bug or a feature request?
Open an [issue](https://github.com/your-repo/sree-assistant/issues) on GitHub.

---

**Built with ❤️ and Python to bring conversations to life.**

