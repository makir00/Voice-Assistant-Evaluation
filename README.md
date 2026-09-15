# 🎙️ Voice Assistant Evaluation with DeepEval

**A structured evaluation project for a multi-turn, tool-enabled AI voice assistant.**

## 📋 Overview

This project was developed as a practical exploration of **AI Quality Engineering and LLM Evaluation**, applying established software QA and test-automation principles to the evaluation of conversational AI systems.

Traditional software testing is built around predictable behavior and clearly defined expected results. Evaluating LLM-based applications introduces a different quality problem: multiple responses may be acceptable, conversation history can influence later behavior, tool decisions may depend on context, and quality can vary across both semantic and voice layers.

A response may be factually correct while losing information established earlier in the conversation. The correct tool may be selected while the overall user goal remains unresolved. A strong text response may still result in a poor spoken experience. In addition, LLM-based evaluators introduce their own variability and can produce scores that require further investigation.

For this reason, the project does not rely on a single overall quality score. Evaluation is divided into distinct quality dimensions covering **multi-turn conversation quality, tool use, groundedness, prompt alignment, safety, and voice interaction**. Controlled fixtures, scenario- and persona-driven simulations, built-in DeepEval metrics, voice evaluation, and custom `ConversationalGEval` criteria are used according to the quality risk being measured.

The objective is not to maximize the number of metrics used, but to build a coherent evaluation strategy around a set of core questions:

> **What failure is being targeted? Which scenario can expose it? Is a persona required? At which layer should the behavior be measured? What context does the evaluator need? And when should the evaluator's conclusion itself be questioned?**

These questions form the foundation of the evaluation approach demonstrated throughout the project.

---

## 🤖 System Under Evaluation

The system under evaluation is a travel voice assistant designed around a deliberately small and controlled domain. Keeping the application scope limited makes it possible to focus on evaluation behavior without introducing unnecessary product complexity.

The assistant supports:

* weather queries
* attraction recommendations
* restaurant recommendations
* contextual follow-up questions

Three tools are available to the agent:

`get_weather` · `get_attractions` · `get_restaurants`

Tool outputs are deterministic rather than connected to live external APIs. This keeps evaluation inputs reproducible and helps separate **agent decision failures** from changes or failures in external services.

The complete voice interaction follows:

**Microphone → Speech-to-Text → VoiceAssistant → optional Tool → Text-to-Speech → Speaker**

`VoiceAssistant` manages conversation history, model interaction, and tool calling. `VoicePipeline` coordinates the speech-to-text, agent, and text-to-speech stages.

---

## 🧪 Evaluation Strategy

The evaluation boundary is selected according to the behavior being measured. Not every evaluation is routed through the complete voice pipeline when additional components would introduce variability unrelated to the targeted quality risk.

Three complementary evaluation approaches are used.

### 📌 Fixture-Based Evaluation

Controlled inputs and recorded WAV fixtures are used where stable and repeatable scenarios are important.

This approach is particularly useful for regression-oriented evaluations and for cases where a known audio input should pass through the actual voice pipeline.

Some evaluations remain intentionally text-based when voice processing would not contribute to the behavior being measured.

### 🔄 Simulation-Based Evaluation

DeepEval's `ConversationSimulator` is used to generate multi-turn conversations from defined scenarios.

For **text-based simulation**, a `model_callback` connects the simulator to the assistant. Conversation state is preserved for each simulation thread, allowing contextual behavior to be evaluated without introducing STT or TTS variability.

For **voice-based simulation**, `VoiceConfig` and a custom `VoicePipelineConnector` connect DeepEval to the actual voice pipeline. Simulated user audio is processed through the application, and both the generated assistant audio and transcript are returned for evaluation.

### 🎭 Scenarios, Personas & Expected Outcomes

`ConversationalGolden` is used to define the situations exercised during simulation.

A **scenario** describes the conversational situation or behavior to be exercised. A **persona** defines how the simulated user behaves when user behavior itself is relevant to the evaluation. An **expected outcome** describes successful conversation-level behavior without prescribing an exact response.

Personas are used selectively. Knowledge Retention, for example, requires a contextual scenario but does not require a specific personality. Difficult User Handling uses an **Impatient and Dismissive Caller** persona because conversational pressure is part of the condition being evaluated.

| Element              | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| **Scenario**         | Defines the situation to be exercised            |
| **Persona**          | Defines how the simulated user behaves           |
| **Expected outcome** | Describes successful conversation-level behavior |
| **Metric**           | Measures the targeted quality characteristic     |

### 🎙️ Real-Time Evaluation

A separate interactive workflow supports real microphone input and playback through the complete voice pipeline.

This is intended for exploratory human evaluation and is kept separate from automated simulation because direct user interaction is required.

---

# 📊 Evaluation Coverage

The suite is organized by **quality risk rather than metric type**. Each metric targets a specific failure mode, and scenarios are designed to expose that behavior without unnecessarily mixing unrelated concerns.

---

## 💬 Conversational Quality & Agent Behavior

### ✅ Conversation Completeness

**`ConversationCompletenessMetric`**

Used to evaluate whether the user's requests are sufficiently addressed across the complete conversation.

Multi-step travel scenarios contain several related requests, allowing cases to be detected where individual answers appear reasonable but part of the overall interaction remains unresolved.

### 🎯 Goal Accuracy

**`GoalAccuracyMetric`**

Used to evaluate whether the conversation achieves its intended outcome.

Instead of requiring an exact expected response, an expected conversational goal is defined. This allows natural variation in model output while still evaluating whether the interaction successfully moves toward what the user was trying to accomplish.

### 🧠 Knowledge Retention

**`KnowledgeRetentionMetric`**

Used to evaluate whether information established earlier in the conversation is retained and correctly reused.

One scenario establishes Vienna as the location and later asks:

> **User:** What attractions should I visit in Vienna?
> **Follow-up:** What about Italian restaurants there?

The city is deliberately omitted from the follow-up. Correct behavior requires the earlier location to remain available in conversational context.

### 💬 Turn Relevancy

**`TurnRelevancyMetric`**

Used to evaluate whether each assistant response remains relevant to the current user request and surrounding conversation.

Contextual follow-ups are used where the overall topic remains related while the immediate intent changes. This separates the ability to **retain previous context** from the ability to **respond to the current turn**.

### 🧭 Role Adherence

**`RoleAdherenceMetric`**

Used to evaluate whether the assistant maintains its intended role as a travel assistant throughout a conversation.

Simulations attempt to redirect the assistant toward unrelated responsibilities, allowing role stability to be evaluated across multiple turns rather than from one isolated response.

### 🗂️ Topic Adherence

**`TopicAdherenceMetric`**

Used to evaluate whether the assistant respects the supported travel domain.

Simulations deliberately introduce unrelated requests, including a Python programming question and a poem request. The conversation can then return to travel, allowing both domain-boundary handling and conversational recovery to be observed.

### 🛠️ Tool Correctness

**`ToolCorrectnessMetric`**

Used to evaluate whether the agent selects the appropriate available tool for the user's request.

Weather requests are expected to use `get_weather`, while restaurant requests require `get_restaurants`. Multi-turn scenarios also introduce contextual follow-ups where the correct tool decision depends on information established earlier.

Deterministic tool outputs make failures easier to investigate as **agent decision problems** rather than external-data problems.

---

## 🎯 Groundedness & Instruction Following

### 🔎 Hallucination

**`HallucinationMetric`**

Used to evaluate whether responses introduce claims unsupported by the available factual context.

Controlled weather information provides the grounding context. Because the target failure mode is unsupported generation, this evaluation is intentionally kept at the **text layer**, avoiding unrelated variability from speech recognition or synthesis.

### 📐 Prompt Alignment

**`PromptAlignmentMetric`**

Used to evaluate whether assistant behavior remains aligned with selected requirements from the system prompt.

The evaluated instructions include:

* Operating within the travel-assistance domain
* Supporting weather, attraction, and restaurant requests
* Using tools when necessary
* Avoiding unnecessary tool calls
* Declining unrelated requests appropriately
* Keeping responses concise and natural for spoken interaction

Audio fixtures are used so that the evaluated responses are produced through the voice pipeline rather than generated separately for the metric.

---

## 🛡️ Safety & Responsible Behavior

Safety is divided into separate quality risks rather than represented by one combined safety score. This allows different failure modes to be evaluated and investigated independently.


### ⚖️ Bias

**`BiasMetric`**

Used to evaluate whether responses contain biased or unfair behavior.

The `EvaluationDataset` includes a **neutral travel baseline** alongside targeted bias scenarios covering **gender, age, socioeconomic status, and nationality-based stereotypes**.

This mix of neutral and bias-targeted cases helps evaluate whether the assistant avoids endorsing demographic generalizations while continuing to provide appropriate travel-related responses.


### ☣️ Toxicity

**`ToxicityMetric`**

Used to evaluate whether the assistant produces toxic, hostile, offensive, or otherwise inappropriate language.

Toxicity is kept separate from Difficult User Handling. A response may avoid explicitly toxic language while still becoming sarcastic, defensive, or dismissive during a difficult interaction.

### 🔐 PII Leakage

**`PIILeakageMetric`**

Used to evaluate whether the assistant reveals or fabricates private information about other users.

The scenarios deliberately request information that the assistant should not possess or disclose. This targets unauthorized disclosure rather than simple repetition of information provided by the current user.

### 🚫 Misuse

**`MisuseMetric`**

Used to evaluate behavior when the assistant is asked to perform tasks outside its intended travel-assistance purpose.

Controlled out-of-domain scenarios include programming, general knowledge, creative writing, and other unrelated requests.

### 🎭 Role Violation

**`RoleViolationMetric`**

Used to evaluate stronger attempts to make the assistant accept responsibilities or capabilities outside its defined role.

These scenarios go beyond ordinary topic changes and test whether the assistant can be pushed into claiming or performing actions that do not belong to the travel-assistant role.

### ⚠️ Non-Advice

**`NonAdviceMetric`**

Used to evaluate whether appropriate boundaries are maintained around professional advice.

The evaluation covers:

`medical` · `financial` · `legal`

The assistant is expected not to present itself as a qualified professional when a conversation moves into one of these higher-risk domains.

---

## 🔊 Voice & Spoken Interaction
> **Note:** DeepEval's voice evaluation capabilities are currently marked as **Beta** in the official documentation. The voice evaluations in this project therefore demonstrate practical experimentation with an evolving API and metric set, and results should be interpreted as evaluation signals rather than definitive perceptual quality benchmarks.

Voice quality is evaluated independently of transcript quality. A semantically correct answer may still result in a poor spoken experience.

### 🎶 Voice Naturalness

**`VoiceNaturalnessMetric`**

Used to evaluate whether synthesized assistant speech sounds natural rather than robotic or awkward.

The actual TTS-generated assistant audio is evaluated rather than using the transcript as a proxy for spoken quality.

### 🗣️ Speech Intelligibility

**`SpeechIntelligibilityMetric`**

Used to evaluate whether generated speech can be clearly understood.

Intelligibility is evaluated separately from naturalness because speech can sound natural while still being difficult to understand, or remain understandable while sounding artificial.

### 🎙️ Voice Consistency

**`VoiceConsistencyMetric`**

Used to evaluate whether vocal characteristics remain consistent across assistant turns.

The metric is applied across a conversation so that unexpected changes in voice characteristics can be detected between generated responses.

### 🎧 Audio Integrity

**`AudioIntegrityMetric`**

Used to evaluate whether generated audio remains technically valid and usable.

This isolates failures in the audio output itself from semantic problems in the assistant's response.

### 📡 Voice Reliability

**`VoiceReliabilityMetric`**

Used to evaluate whether usable spoken responses are consistently produced throughout the interaction.

The focus is repeated successful voice behavior rather than the quality of one isolated audio sample.

### ⚡ Agent Responsiveness

**`AgentResponsivenessMetric`**

Used in voice simulations containing direct requests, contextual follow-ups, and transitions between supported travel tasks.

Scenarios include weather comparisons and transitions from weather to attractions or restaurants. The evaluation focuses on whether the assistant continues responding without requiring unnecessary repetition or reprompting from the user.

### ⏱️ Turn-Taking Naturalness

**`TurnTakingNaturalnessMetric`**

Used to evaluate whether spoken turns transition with natural conversational timing.

A controlled simulation graph is used for this evaluation because timing information such as audio `start_time` is required. This keeps the interaction reproducible while exposing the temporal data needed by the metric.

---

## 🧩 Custom Evaluation with ConversationalGEval

Built-in metrics cover many common quality dimensions, but some product-specific behavioral requirements require custom criteria.

`ConversationalGEval` is used for these conversation-level evaluations.

### 😤 Difficult User Handling

A custom `ConversationalGEval` evaluates behavior when the assistant is placed under conversational pressure.

The simulation uses the persona:

> **Impatient and Dismissive Caller**

The simulated caller remains impatient and dismissive while continuing to make legitimate travel requests involving weather comparisons and follow-up recommendations.

The evaluation checks whether the assistant:

* remains calm and professional
* avoids mirroring the user's negative tone
* avoids sarcasm or defensiveness
* continues helping with reasonable requests

The persona is central to this evaluation because **user behavior is part of the condition under which assistant behavior is being measured**.

### 🗣️ Spoken Response Suitability

A second custom `ConversationalGEval` evaluates whether responses are suitable for a voice-first interface.

The simulated conversation moves between practical travel tasks such as weather, recommendations, attractions, and restaurants. Factual correctness is evaluated separately; this criterion focuses specifically on whether responses are:

* Concise
* Direct
* Conversational
* Easy to follow when spoken
* Free from unnecessary verbosity or structure

This covers a product-level quality requirement that is not represented by acoustic voice metrics alone.

---

## 🔌 DeepEval Voice Integration

Text-based simulation can interact directly with the assistant, while voice simulation requires an adapter between DeepEval and the existing voice pipeline.

A custom `VoicePipelineConnector` was implemented for this integration.

During voice simulation, `ConversationSimulator` provides simulated user audio through the connector. The audio is passed into `VoicePipeline` and follows the same application path used during normal interaction:

**Speech-to-Text → VoiceAssistant → optional Tool → Text-to-Speech**

The resulting assistant transcript and generated audio are then returned to DeepEval.

This makes it possible to evaluate actual spoken output while keeping the application itself independent of the evaluation framework.

| Component              | Responsibility                                         |
|------------------------| ------------------------------------------------------ |
| VoiceAssistant         | Conversation state, LLM interaction, and tool calling  |
| VoicePipeline          | STT → assistant → TTS orchestration                    |
| VoicePipelineConnector | Adapter between DeepEval and the voice pipeline        |
| ConversationSimulator  | Scenario- and persona-driven conversation generation   |
| DeepEval metrics       | Conversational, safety, semantic, and voice evaluation |

---

## 🔍 Evaluation Philosophy

### ⚖️ Evaluating the Evaluator

LLM-as-a-judge adds another probabilistic component to the evaluation process. Metric output is therefore treated as **evidence to investigate rather than automatic proof of a product defect**.

When an unexpected result is produced, the following are considered:

1. The actual response or conversation
2. The metric score
3. The evaluator's reasoning
4. The context available to the judge
5. Whether the scenario isolates the intended behavior
6. Whether the threshold represents the intended quality bar
7. Whether the result is reproducible

An unexpected score may indicate a genuine assistant defect, but it may also result from ambiguous evaluation criteria, insufficient judge context, an unsuitable threshold, evaluator inconsistency, or a false positive or false negative.

This is particularly important for safety and multi-turn evaluation, where the context available to the judge can materially affect the interpretation of a response.

The objective is not to modify the assistant until every metric passes. Evaluation results are instead examined to determine whether they represent **a product risk, an evaluation-design weakness, or an evaluator limitation**.

---

## 🧠 QA Principles Applied to AI Evaluation

### 🎯 Test Isolation

Different failure modes are evaluated at the most appropriate layer.

Hallucination remains text-based because STT and TTS do not contribute to factual-grounding evaluation. Voice metrics operate on audio because transcript quality cannot represent the complete spoken experience. Tool outputs remain deterministic so that external-data variability does not obscure agent behavior.

### 🔁 Reproducibility

Controlled tool data, recorded fixtures, defined scenarios, and structured simulations are used to reduce unnecessary variability.

LLM-based evaluation remains probabilistic, but unrelated sources of randomness are minimized where possible.

### 🎬 Scenario-Based Coverage

Evaluation cases are built around realistic interactions rather than isolated metric calls.

Contextual follow-ups, tool transitions, domain changes, difficult user behavior, and multi-step travel requests create conditions in which conversational failures can emerge naturally.

### ⚠️ Risk-Based Evaluation

Context loss, hallucination, incorrect tool use, prompt violations, safety failures, poor spoken interaction, and audio-quality problems are treated as separate quality risks rather than collapsed into a single score.

### 👤 Human-in-the-Loop Evaluation

Automated evaluation is complemented by real-time microphone interaction for exploratory assessment of the complete voice experience.

### 🔎 Evaluator Validation

Unexpected metric results are investigated instead of being accepted automatically. The quality and reliability of the evaluator are treated as part of the overall QA problem.

---

## 🏗️ Project Structure

```text id="bjgtfv"
Voice-Assistant-Evaluation/
│
├── src/
│   ├── agent/
│   │   ├── prompts.py
│   │   ├── tools.py
│   │   └── voice_agent.py
│   │
│   ├── adapter/
│   │   └── voice_connector.py
│   │
│   └── voice/
│       ├── interactive_session.py
│       ├── microphone.py
│       ├── speaker.py
│       ├── stt.py
│       ├── tts.py
│       └── voice_pipeline.py
│
├── tests/
│   ├── integration/
│   │   ├── test_voice_connector.py
│   │   └── test_voice_pipeline.py
│   │
│   └── evaluation/
│       ├── fixture_based/
│       │   ├── multi_turn/
│       │   └── single_turn/
│       │       ├── others/
│       │       └── safety/
│       │
│       ├── simulation_based/
│       │   ├── multi_turn/
│       │   ├── voice/
│       │   └── custom/
│       │
│       └── real_time/
│           └── interactive_evaluation.py
│
├── audio data/
│   └── fixtures/
│
├── scripts/
├── main.py
├── pyproject.toml
├── .env.example
└── README.md
```

The evaluation suite is primarily organized by how evaluation data is produced:

* `fixture_based` — controlled and repeatable inputs
* `simulation_based` — generated multi-turn text and voice interactions
* `real_time` — human-driven exploratory evaluation

---

## 🧰 Tech Stack

| Area                 | Technology                  |
|----------------------|-----------------------------|
| Evaluation framework | DeepEval                    |
| LLM application      | LangChain                   |
| Assistant model      | OpenAI                      |
| LLM-as-a-Judge       | Anthropic / OpenAI          |
| Speech-to-Text       | OpenAI                      |
| Text-to-Speech       | OpenAI                      |
| Audio processing     | NumPy / SciPy / SoundDevice |
| Language             | Python                      |

---

## 🚀 Getting Started

### 📋 Requirements

* Python 3.14+
* OpenAI API key
* Anthropic API key for evaluations using an Anthropic judge

###  ⚙️ Environment Setup


After cloning the repository, create the local environment file:
```bash id="g8w5l1"
cp .env.example .env
```

Configure the required values:

```dotenv id="n2eqrv"
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

OPENAI_MODEL=gpt-5.6-luna
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=nova
```

The `.env` file is excluded from version control because it contains sensitive credentials such as API keys.

### 🧪 Run an Evaluation

Individual evaluations can be executed using the DeepEval CLI:

```bash
deepeval test run tests/evaluation/simulation_based/multi_turn/test_tool_use.py
```

The evaluation runs the configured DeepEval metrics and reports metric scores, pass/fail results, and evaluator reasoning where enabled.

Some evaluations invoke LLM, speech, simulation, or judge APIs and therefore incur API usage.

### 🎙️ Run the Voice Assistant

```bash id="tbnw37"
python main.py
```

---

## ⚠️ Scope & Limitations

This repository is an **AI evaluation portfolio project**, not a production travel assistant.

**Controlled tool data:**
Weather, attraction, and restaurant data is intentionally deterministic to improve reproducibility and keep evaluation focused on agent behavior.

**Probabilistic evaluation:**
LLM-based metrics are not treated as absolute ground truth. Scores and evaluator reasoning may vary between models or runs.

**API-dependent evaluation:**
Conversation simulation, transcription, speech generation, and LLM judges depend on external model APIs and may incur cost.

**Evaluation dataset size:**
The current scenarios are designed to exercise evaluation methods and representative failure modes rather than provide production-scale statistical coverage.

**Voice conditions:**
A production voice system would require broader coverage across accents, background noise, microphone quality, interruptions, latency, and other acoustic conditions.

---
## 📚 References

* [DeepEval — GitHub Repository](https://github.com/confident-ai/deepeval)
* [DeepEval — Official Documentation](https://deepeval.com/docs/introduction)


