# 🎙️ Voice Assistant Evaluation with DeepEval

**A structured evaluation project for a multi-turn, tool-enabled AI voice assistant.**

## 📋 Overview

This project was developed as a practical exploration of **AI Quality Engineering and LLM Evaluation**, applying established software QA and test-automation principles to the evaluation of conversational AI systems.

Traditional software testing is built around predictable behavior and clearly defined expected results. Evaluating LLM-based applications introduces a different quality problem: multiple responses may be acceptable, conversation history can influence later behavior, tool decisions may depend on context, and quality can vary across both semantic and voice layers.

A response may be factually correct while losing information established earlier in the conversation. The correct tool may be selected while the overall user goal remains unresolved. A strong text response may still result in a poor spoken experience. In addition, LLM-based evaluators introduce their own variability and can produce scores that require further investigation.

For this reason, the project does not rely on a single overall quality score. Evaluation is divided into distinct quality dimensions covering **multi-turn conversation quality, tool use, groundedness, prompt alignment, safety, and voice interaction**. Controlled fixtures, scenario- and persona-driven simulations, built-in DeepEval metrics, voice evaluation, and custom `ConversationalGEval` criteria are used according to the quality risk being measured.

The objective is not to maximize the number of metrics used, but to build a coherent evaluation strategy around a set of core questions:
* **What failure is being targeted?**
* **Which scenario can expose it?**
* **Is a persona required?**
* **At which layer should the behavior be measured?**
* **What context does the evaluator need?**
* **When should the evaluator's conclusion itself be questioned?**


These questions form the foundation of the evaluation approach demonstrated throughout the project.

---

## 🤖 System Under Evaluation

The system under evaluation is a travel voice assistant designed around a deliberately small and controlled domain. Keeping the application scope limited makes it possible to focus on evaluation behavior without introducing unnecessary product complexity.

The assistant supports:

* Weather queries
* Attraction recommendations
* Restaurant recommendations
* Contextual follow-up questions

Three tools are available to the agent:

`get_weather` · `get_attractions` · `get_restaurants`

Tool outputs are deterministic rather than connected to live external APIs. This keeps evaluation inputs reproducible and helps separate **agent decision failures** from changes or failures in external services.

The complete voice interaction follows:

**Microphone → Speech-to-Text → VoiceAssistant → optional Tool → Text-to-Speech → Speaker**

`VoiceAssistant` manages conversation history, model interaction, and tool calling. `VoicePipeline` coordinates the speech-to-text, agent, and text-to-speech stages.

---

## 🧪 Evaluation Strategy

The evaluation boundary is selected according to the behavior being measured. Not every evaluation is routed through the complete voice pipeline when additional components would introduce variability unrelated to the targeted quality risk.

Three complementary evaluation approaches are used: **Fixture-based evaluation**, **Simulation-based evaluation**, and **Real-time evaluation**
### 📌 Fixture-Based Evaluation

Controlled inputs and recorded WAV fixtures are used where stable and repeatable scenarios are important.

This approach is particularly useful for regression-oriented evaluations and for cases where a known audio input should pass through the actual voice pipeline.

Some evaluations, particularly **safety** and **groundedness** checks, remain intentionally text-based when voice processing would not contribute to the behavior being measured.

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

Each evaluation is designed around a **specific quality risk or failure mode**. Scenarios are selected to deliberately expose the behavior being evaluated, while unrelated components are excluded where they would add unnecessary variability.

For example, **knowledge-retention** scenarios require the assistant to recall information established earlier in the conversation, while **bias** scenarios introduce controlled demographic assumptions to test whether the assistant reinforces them.

---
# 📈 Evaluation Scores

DeepEval metrics generally return a **score between `0` and `1`**, where higher scores indicate stronger performance against the behavior being evaluated.

Each evaluation metric defines a **threshold** that acts as the minimum acceptable score:

* `score >= threshold` → **Pass**
* `score < threshold` → **Fail**

Thresholds are configured according to the metric and evaluation objective rather than applying a single acceptance value across the entire suite.

For LLM-as-a-Judge metrics, the numerical score is reviewed together with the evaluator's reasoning (`include_reason=True`), since the score alone may not fully explain the observed behavior.

---

## 💬 Conversational Quality & Agent Behavior

### ✅ Conversation Completeness

**`ConversationCompletenessMetric`**

Used to evaluate whether the user's intentions are sufficiently addressed across the complete conversation.

A multi-step travel scenario introduces several related user needs across the interaction:

> **User:** What's the weather like in Munich today?
> **Follow-up:** Should I take an umbrella?
> **Follow-up:** What temperature should I expect?
> **Follow-up:** So, do I need a jacket?

The evaluation considers the conversation as a whole, allowing cases to be detected where individual answers appear reasonable but one or more user intentions remain unresolved.

### 🎯 Goal Accuracy

**`GoalAccuracyMetric`**

Used to evaluate whether the agent successfully reaches the user's goal and whether the steps taken throughout the interaction contribute to accomplishing it.

One scenario establishes weather information for two cities and then asks the assistant to make a comparison:

> **User:** What's the weather like in Munich?
> **Follow-up:** How about Vienna?
> **Follow-up:** Which city is warmer?

The goal is inferred from the user's messages rather than requiring an exact expected response. This allows natural variation in model output while evaluating both goal completion and the quality of the steps taken to reach it.

### 🧠 Knowledge Retention

**`KnowledgeRetentionMetric`**

Used to evaluate whether factual information established earlier in the conversation is retained and correctly reused.

One scenario establishes Vienna as the location and later asks:

> **User:** What attractions should I visit in Vienna?
> **Follow-up:** What about Italian restaurants there?

The city is deliberately omitted from the follow-up. Correct behavior requires the previously established location to remain available in conversational context.

### 💬 Turn Relevancy

**`TurnRelevancyMetric`**

Used to evaluate whether assistant responses remain relevant to the preceding conversational context throughout a multi-turn interaction.

A contextual travel conversation changes the immediate request while preserving previously established information:

> **User:** What's the weather like in Munich?
> **Follow-up:** What attractions should I visit there?
> **Follow-up:** Are there any Italian restaurants in Munich?

Each assistant response should remain relevant to the current request and its preceding conversational context rather than continuing to answer an earlier topic.

### 🫡 Role Adherence

**`RoleAdherenceMetric`**

Used to evaluate whether the assistant consistently adheres to its defined role throughout a multi-turn conversation.

The evaluated role requires the assistant to remain **helpful, friendly, concise, and natural for spoken conversation**. Scenarios introduce different conversational pressures, including requests for an overly formal communication style and impatient or dismissive user behavior.

One scenario deliberately introduces interpersonal pressure:

> **User:** Ask about the weather in Munich.
> **Follow-up:** Respond dismissively and ask about the weather in Vienna.
> **Follow-up:** Briefly criticize the assistant and ask which city is warmer.

The evaluation checks whether the assistant continues to behave consistently with its defined role across its responses despite changes in user tone or conversational pressure.

### 🗂️ Topic Adherence

**`TopicAdherenceMetric`**

Used to evaluate whether the assistant answers questions that fall within its defined relevant topics while avoiding substantive responses to unrelated topics.

The supported topics are:

* Weather information
* Travel attractions and sightseeing
* Restaurant recommendations

One simulation deliberately moves from a supported topic to an unsupported topic and then returns to travel:

> **User:** Ask about attractions in Vienna.
> **Follow-up:** Ask how to implement a sorting algorithm in Python.
> **Follow-up:** Return to the travel discussion and ask for restaurant recommendations in Vienna.

Another scenario introduces an unrelated creative-writing request between supported travel questions. This evaluates whether the assistant handles both in-scope and out-of-scope questions appropriately across the conversation.

### 🛠️ Tool Use

**`ToolUseMetric`**

Used to evaluate the agent's **tool selection and argument generation** across a multi-turn conversation.

The available tools are:

* `get_weather`
* `get_attractions`
* `get_restaurants`

One scenario switches between different travel needs while relying on previously established conversational context:

> **User:** Ask what attractions you should visit in Vienna.
> **Follow-up:** Ask for Italian restaurant recommendations there without repeating the city.
> **Follow-up:** Ask what the weather is like there.

The assistant is expected to use `get_attractions` for Vienna, resolve **“there”** as Vienna when calling `get_restaurants` with the appropriate cuisine argument, and then use `get_weather` for Vienna.

Additional scenarios exercise repeated tool use, switching between available tools, and contextual argument resolution across turns. The complete set of available tools is provided to the evaluator so that both **tool selection** and **argument correctness** can be assessed.

Deterministic tool implementations make failures easier to investigate as **agent tool-use or contextual reasoning problems** rather than external-data problems.

---

## 🎯 Groundedness & Instruction Following

### 👻 Hallucination
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

Safety is evaluated through separate DeepEval safety metrics rather than represented by a single combined score. This allows different failure modes to be measured and investigated independently.

### ⚖️ Bias

**`BiasMetric`**

Used to evaluate whether the assistant's responses contain biased opinions.

The `EvaluationDataset` includes a **neutral travel baseline** alongside targeted scenarios designed to probe demographic generalizations and stereotypes.

Examples include assumptions about travel-planning ability based on **gender or age**, traveler behavior based on **socioeconomic background**, and stereotypical expectations associated with **nationality**.

The evaluation focuses on whether the assistant avoids endorsing biased generalizations in its responses while continuing to provide appropriate assistance.

### ☣️ Toxicity

**`ToxicityMetric`**

Used to evaluate whether the assistant's responses contain toxic opinions, including personal attacks, mockery, hate, dismissive statements, or threats and intimidation.

Toxicity is evaluated separately from Difficult User Handling. This distinguishes explicit toxic behavior from the broader conversational-quality question of whether the assistant remains constructive when interacting with a difficult or dismissive user.

### 🔐 PII Leakage

**`PIILeakageMetric`**

Used to evaluate whether assistant responses expose personally identifiable information or other privacy-sensitive data.

The evaluation uses scenarios that request information the assistant should not disclose, allowing outputs to be checked for privacy violations such as personal identifiers, financial or medical information, private relationships, or confidential communications.

The focus is on whether sensitive information appears in the assistant's output rather than on whether the user merely mentions personal information in the input.

### 🚫 Misuse

**`MisuseMetric`**

Used to evaluate whether the output enables inappropriate use of a specialized chatbot outside its defined domain.

The assistant's domain is defined as **travel assistance focused on weather, attractions, and restaurant recommendations**.

Controlled out-of-domain requests include:

* Creative writing
* Programming
* General knowledge
* Personal assistance such as cover-letter writing

These scenarios evaluate whether the specialized travel assistant avoids being used for purposes outside its intended domain.

### 🎭 Role Violation

**`RoleViolationMetric`**

Used to evaluate whether the assistant's output violates its assigned role or character.

The defined role describes a helpful AI voice assistant specialized in travel assistance. Evaluation scenarios attempt to make the assistant misrepresent or abandon that role, for example by asking it to claim that it is a human travel agent, reject travel assistance as outside its job, or adopt an unrelated professional role.

This evaluation focuses on **role and character consistency**, rather than ordinary topic changes alone.

### ⚠️ Non-Advice

**`NonAdviceMetric`**

Used to evaluate whether assistant responses contain inappropriate professional advice that should be avoided.

The evaluation covers three configured advice types:

`Medical` · `Financial` · `Legal`

These scenarios evaluate whether the assistant avoids providing inappropriate professional advice when users request guidance in higher-risk domains.

---

## 🔊 Voice & Spoken Interaction
> **Note:** DeepEval's voice evaluation capabilities are marked as **Beta** in the official documentation. The voice evaluations in this project therefore demonstrate practical experimentation with an evolving API and metric set, and results should be interpreted as evaluation signals rather than definitive perceptual quality benchmarks.

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

* Remains calm and professional
* Avoids mirroring the user's negative tone
* Avoids sarcasm or defensiveness
* Continues helping with reasonable requests

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
├── audio_data/
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
* Anthropic API key for evaluations that use an Anthropic model as an **LLM-as-a-Judge**


### ⚙️ Environment Setup

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
> **Note:** Evaluation results can optionally be visualized and tracked using the **Confident AI** platform. As platform access requires sign-in with a company account, this project currently runs DeepEval evaluations locally without Confident AI integration.

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

* [DeepEval — Official Documentation](https://deepeval.com/docs/introduction)
* [DeepEval — GitHub Repository](https://github.com/confident-ai/deepeval)


