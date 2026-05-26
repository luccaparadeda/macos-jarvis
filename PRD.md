That makes complete sense. Stripping away the external smart home complexity keeps this project hyper-focused on what matters right now: building a lightning-fast, native macOS assistant that perfectly manages your immediate digital and physical workspace.

By removing the network-based controls (Firestick and Google Home), we drastically simplify the Python orchestration layer and reduce potential points of failure, letting you master the core loop of **Wake Word -> Listen -> Look -> Think -> Run Shortcut -> Speak**.

Plus, with DeepSeek recently releasing the **V4 series** (specifically `deepseek-v4-flash` with its massive 1M context window and enhanced agentic tool calling), the cloud brain is going to handle those macOS Shortcuts effortlessly while keeping costs at virtually zero.

Here is your streamlined, laser-focused PRD.

---

## **Product Requirements Document (PRD)**

**Project Name:** Jarvis (Streamlined macOS Spatial Assistant)
**Document Status:** Final Draft
**Target Platform:** macOS (Apple Silicon M1-M4, specifically optimized for 8GB Unified Memory)

---

### **1. Product Vision & Objective**

**Vision:** To build a highly responsive, context-aware voice assistant embedded deeply into the native macOS ecosystem. It acts as a true spatial assistant by utilizing the user's iPhone as its eyes and Apple Shortcuts as its primary digital hands, ignoring external smart home complexities in favor of absolute desktop mastery.
**Objective:** Architect a hybrid pipeline that uses ultra-lightweight local models for audio processing (wake word, transcription, synthesis) to maintain low latency, while offloading heavy visual reasoning and tool routing to a fast, cost-effective cloud API (DeepSeek V4). The architecture must include a zero-friction toggle to transition to a 100% local offline setup in the future.

---

### **2. Target Architecture & Tech Stack**

This modular stack is strictly designed to idle near 0MB of RAM and peak well under the 5GB usable limit of an 8GB Apple Silicon Mac.

| Component | Technology | Primary Function | Execution Environment |
| --- | --- | --- | --- |
| **Trigger** | Picovoice Porcupine | Always-on, low-power wake word detection ("Jarvis"). | 100% Local |
| **Ears (STT)** | MLX-Audio / Whisper-Tiny | Transcribes user audio commands into text instantly. | 100% Local |
| **Eyes (Vision)** | OpenCV / Continuity Camera | Captures high-res image frames from a linked iPhone. | 100% Local |
| **Brain (LLM)** | DeepSeek V4-Flash (via OpenAI SDK) | Processes text/images, maintains context, and routes tools. | Cloud API (Future: Local Ollama) |
| **Hands (macOS)** | Python `subprocess` | Executes native macOS Apple Shortcuts. | 100% Local |
| **Mouth (TTS)** | MLX-Audio / Kokoro | Synthesizes text back into natural human speech. | 100% Local |

---

### **3. Functional Requirements**

These are the core capabilities the system must execute reliably for the Minimum Viable Product (MVP).

* **Wake Word Activation:** The system must listen efficiently in the background and only activate the main pipeline when the keyword "Jarvis" is detected.
* **Intelligent Visual Capture:** The system must automatically trigger the macOS Continuity Camera via OpenCV to snap and base64-encode a photo if the user's prompt implies visual context (e.g., "look at," "see," "what is this").
* **macOS System Control:** The system must be able to trigger native Apple Shortcuts by name, passing optional text inputs, to manipulate desktop apps, write notes, send iMessages, or manage local calendar events.
* **API Standardization:** The orchestrator must utilize the standard OpenAI Python SDK format (pointed to `https://api.deepseek.com`) to ensure the cloud provider can be instantly swapped for a local provider (Ollama) by changing a single URL string.
* **Audio Playback:** Text output from the LLM must stream into the Kokoro TTS engine and play through the default Mac speakers.

---

### **4. Non-Functional Requirements**

These requirements dictate the quality, performance, and strict hardware constraints of the system.

* **Memory Management:** The combined active footprint of Whisper, Kokoro, and the Python orchestrator must not exceed 2.5GB of unified memory to prevent macOS from swapping memory to the SSD.
* **Latency Tolerances:**
* Wake word detection to recording start: < 100ms.
* Voice transcription (STT): < 300ms.
* API Roundtrip (Text only): < 1,000ms.
* API Roundtrip (Vision/Image included): < 3,000ms.
* Apple Shortcut Execution Initiation: < 200ms.


* **Privacy:** All audio processing (listening, transcribing, and speaking) must occur strictly on-device. Only explicit text prompts and captured Continuity Camera frames are transmitted to the cloud API.

---

### **5. User Interaction Flow**

The step-by-step logical sequence the application executes.

1. **Idle State:** Porcupine monitors microphone input. RAM usage is minimal.
2. **Trigger:** User says "Jarvis". System pauses wake word detection and records command until silence.
3. **Transcribe:** Whisper converts the audio buffer to a text string.
4. **Visual Check:** Script evaluates if the user is asking to "look" at something. If yes, it wakes the iPhone, snaps a frame, and releases the camera.
5. **Think:** Text (and optional image) is sent to the DeepSeek V4 API alongside the Shortcut tool schema.
6. **Act (Tool Loop):**
* DeepSeek requests an Apple Shortcut execution.
* Python executes the chosen Shortcut locally via terminal commands and returns the success/failure state to DeepSeek.
* DeepSeek drafts a final spoken response based on the tool's result.


7. **Synthesize:** The text stream flows into Kokoro.
8. **Output:** Mac plays the synthesized audio.
9. **Reset:** System resumes Porcupine wake word detection.

---

### **6. Future Milestones (Post-MVP)**

Features slated for future development once the core loop is stable or hardware is upgraded.

* **100% Local Migration:** Switching the `base_url` to a local Ollama instance running a small vision model (like Moondream or Llama Vision) once a machine with 16GB+ unified memory is acquired.
* **Daemonization:** Wrapping the Python script in a `launchd` plist file so it runs completely hidden in the background the moment the Mac boots up, without needing an open Terminal window.
* **Dynamic Tool Discovery:** Allowing the script to dynamically read the user's Shortcut library at runtime and automatically update the JSON tool schema, rather than hardcoding specific Shortcut names into the script.
