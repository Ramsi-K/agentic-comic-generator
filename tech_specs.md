# Technical Specs

## 🔄 Multi-Turn Agent Communication

### Feedback Loop Implementation

1. **Initial Request**: Brown sends structured prompt to Bayko
2. **Content Generation**: Bayko processes via Modal + sponsor APIs
3. **Quality Validation**: Brown evaluates output against original intent
4. **Iterative Refinement**: Up to 3 feedback cycles with specific improvement requests
5. **Final Assembly**: Brown compiles approved content into deliverable format

### Agent Message Schema

```json
{
  "message_id": "msg_001",
  "timestamp": "2025-01-15T10:30:00Z",
  "sender": "agent_brown",
  "recipient": "agent_bayko",
  "message_type": "generation_request",
  "payload": {
    "prompt": "A moody K-pop idol finds a puppy",
    "style_tags": ["studio_ghibli", "whisper_soft_lighting"],
    "panels": 4,
    "language": "korean",
    "extras": ["narration", "subtitles"]
  },
  "context": {
    "conversation_id": "conv_001",
    "iteration": 1,
    "previous_feedback": null
  }
}
```

---

## 📁 File Organization & Data Standards

### Output Directory Structure

```text
/storyboard/
├── session_001/
│   ├── agents/
│   │   ├── brown_state.json      # Agent Brown memory/state
│   │   ├── bayko_state.json      # Agent Bayko memory/state
│   │   └── conversation_log.json # Inter-agent messages
│   ├── content/
│   │   ├── panel_1.png          # Generated images
│   │   ├── panel_1_audio.mp3    # TTS narration
│   │   ├── panel_1_subs.vtt     # Subtitle files
│   │   └── metadata.json        # Content metadata
│   ├── iterations/
│   │   ├── v1_feedback.json     # Validation feedback
│   │   ├── v2_refinement.json   # Refinement requests
│   │   └── final_approval.json  # Final validation
│   └── output/
│       ├── final_comic.png      # Assembled comic
│       ├── manifest.json        # Complete session data
│       └── performance_log.json # Timing/cost metrics
```

### Metadata Standards

```json
{
  "session_id": "session_001",
  "created_at": "2025-01-15T10:30:00Z",
  "user_prompt": "Original user input",
  "processing_stats": {
    "total_iterations": 2,
    "processing_time_ms": 45000,
    "api_calls": {
      "openai": 3,
      "mistral": 2,
      "modal": 8
    },
    "cost_breakdown": {
      "compute": "$0.15",
      "api_calls": "$0.08"
    }
  },
  "quality_metrics": {
    "brown_approval_score": 0.92,
    "style_consistency": 0.88,
    "prompt_adherence": 0.95
  }
}
```

---

## ⚙️ Tool Orchestration & API Integration

### Modal Compute Layer

```python
# Modal function for SDXL image generation
@app.function(
    image=modal.Image.debian_slim().pip_install("diffusers", "torch"),
    gpu="A10G",
    timeout=300
)
def generate_comic_panel(prompt: str, style: str) -> bytes:
    # SDXL pipeline with HuggingFace integration
    return generated_image_bytes
```

### Sponsor API Integration

- **OpenAI GPT-4**: Dialogue generation and character voice consistency
- **Mistral**: Style adaptation and tone refinement
- **HuggingFace**: SDXL model hosting and inference
- **Modal**: Serverless GPU compute for image/audio generation

> Mistral Agents: Investigated experimental client.beta.agents framework for dynamic task routing, but deferred due to limited stability at time of build.

### LlamaIndex Agent Memory

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

# Agent Brown with persistent memory
brown_agent = ReActAgent.from_tools(
    tools=[validation_tool, feedback_tool, assembly_tool],
    memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
    verbose=True
)
```

---

## 🌐 Gradio-FastAPI Integration

### Frontend Architecture

```python
import gradio as gr
from fastapi import FastAPI
import asyncio

app = FastAPI()

# Gradio interface with real-time updates
def create_comic_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Input components
        prompt_input = gr.Textbox(label="Story Prompt")
        style_dropdown = gr.Dropdown(["Studio Ghibli", "Manga", "Western"])

        # Real-time status display
        status_display = gr.Markdown("Ready to generate...")
        progress_bar = gr.Progress()

        # Agent thinking display
        agent_logs = gr.JSON(label="Agent Decision Log", visible=True)

        # Output gallery
        comic_output = gr.Gallery(label="Generated Comic Panels")

        # WebSocket connection for real-time updates
        demo.load(setup_websocket_connection)

    return demo
```

### Real-Time Agent Status Updates

- **Agent Thinking Display**: Live JSON feed of agent decision-making
- **Progress Tracking**: Visual progress bar with stage indicators
- **Error Handling**: Graceful failure recovery with user feedback
- **Performance Metrics**: Real-time cost and timing information

---

## 🚀 Deployment Configuration

### HuggingFace Spaces Frontend

```yaml
# spaces_config.yml
title: Agentic Comic Generator
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: '4.0.0'
app_file: app.py
pinned: false
license: mit
```

### Modal Backend Services

```python
# modal_app.py
import modal

app = modal.App("agentic-comic-generator")

# Shared volume for agent state persistence
volume = modal.Volume.from_name("comic-generator-storage")

@app.function(
    image=modal.Image.debian_slim().pip_install_from_requirements("requirements.txt"),
    volumes={"/storage": volume},
    keep_warm=1
)
def agent_orchestrator():
    # Main agent coordination logic
    pass
```

### Environment Configuration

```python
# config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Sponsor API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY")
    hf_token: str = os.getenv("HF_TOKEN")

    # Modal configuration
    modal_token_id: str = os.getenv("MODAL_TOKEN_ID")
    modal_token_secret: str = os.getenv("MODAL_TOKEN_SECRET")

    # Application settings
    max_iterations: int = 3
    timeout_seconds: int = 300
    debug_mode: bool = False
```

---

## 🔧 Extensibility Framework

### Plugin Architecture

```python
# plugins/base.py
from abc import ABC, abstractmethod

class ContentPlugin(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: dict) -> dict:
        pass

    @abstractmethod
    def validate(self, content: dict) -> bool:
        pass

# plugins/tts_plugin.py
class TTSPlugin(ContentPlugin):
    async def generate(self, text: str, voice: str) -> bytes:
        # TTS implementation using sponsor APIs
        pass
```

### Agent Extension Points

- **Custom Tools**: Easy integration of new AI services
- **Memory Backends**: Swappable persistence layers (Redis, PostgreSQL)
- **Validation Rules**: Configurable content quality checks
- **Output Formats**: Support for video, interactive comics, AR content

### API Abstraction Layer

```python
# services/ai_service.py
class AIServiceRouter:
    def __init__(self):
        self.providers = {
            "dialogue": OpenAIService(),
            "style": MistralService(),
            "image": HuggingFaceService(),
            "compute": ModalService()
        }

    async def route_request(self, service_type: str, payload: dict):
        return await self.providers[service_type].process(payload)
```

---

## 📊 Performance & Monitoring

### Metrics Collection

- **Agent Performance**: Decision time, iteration counts, success rates
- **API Usage**: Cost tracking, rate limiting, error rates
- **User Experience**: Generation time, satisfaction scores
- **System Health**: Resource utilization, error logs

#### Cost Optimization

- **Smart Caching**: Reuse similar generations across sessions
- **Batch Processing**: Group API calls for efficiency
- **Fallback Strategies**: Graceful degradation when services are unavailable

---
