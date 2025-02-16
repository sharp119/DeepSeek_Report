# DeepSeek AI Model Repository

Welcome to the DeepSeek AI Model Repository! This repository serves as a comprehensive resource for understanding the landscape of AI models developed by DeepSeek AI, as of early 2025. It provides overviews, key features, and relevant links for various models across different domains, including Large Language Models (LLMs), code generation, mathematical problem-solving, and vision-language tasks.

## Purpose

The primary goal of this repository is to offer a centralised location for individuals interested in DeepSeek AI's contributions to the field of artificial intelligence. Whether you are a researcher, developer, or simply an AI enthusiast, this repository aims to provide you with a clear and concise overview of DeepSeek AI's model ecosystem.

## Structure

The repository is organised into several sections, each focusing on a specific category of DeepSeek AI models. Each section contains a summary of the models, key architectural details, training information, and links to official resources, such as Hugging Face pages and relevant publications.

The main components of this repository are:

*   **README.md:** This file, providing an overview of the repository's purpose, structure, and contents.
*   **Model-Specific Markdown Files:** Detailed reports on specific DeepSeek AI models, architectures, and functionalities. These files contain in-depth information compiled from various sources, including Hugging Face, research papers, and community discussions.
    *   DeepSeek-LLM.md
    *   DeepSeek-V2.md
    *   DeepSeek-V3.md
    *   DeepSeek-R1.md
    *   DeepSeek-Coder.md
    *   DeepSeekCoder-V2.md
    *   DeepSeek-Math.md
    *   DeepSeek-Prover.md
    *   DeepSeek-VL.md
    *   DeepSeek-VL2.md
    *   DeepSeek-MoE.md
    *   DeepSeek-V2.5.md (Unofficial)

## Model Overview

This section provides a brief summary of the various DeepSeek AI models covered in this repository. For more detailed information, please refer to the individual model-specific markdown files.

### Large Language Models (LLMs)

#### DeepSeek-LLM Family

**"DeepSeek-LLM" is a general term referring to DeepSeek AI's text-based Large Language Models** [1, 2]. It encompasses models designed for general language understanding, generation, and reasoning [1, 2]. Key characteristics include a transformer architecture, large scale, and a focus on efficiency [2].

*   **Included Models:** DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, and potentially future text-based models [1, 2].

#### Key LLMs:

*   **DeepSeek-V2**
    *   A 236 billion parameter Mixture-of-Experts (MoE) model, with 21 billion parameters active per token [3, 4].
    *   Employs Multi-head Latent Attention (MLA) and Grouped Query Attention (GQA) for efficiency [3].
    *   Trained on 8.1 trillion tokens [3].
    *   Known for its strong performance-to-cost ratio [3].
    *   [Hugging Face Page](DeepSeek-V2 Hugging Face Page)
*   **DeepSeek-V3**
    *   A 671 billion parameter MoE model, with 37 billion parameters active per token [5-7].
    *   Utilises MLA and a novel auxiliary-loss-free strategy for load balancing [5-7].
    *   Trained using a Multi-Token Prediction (MTP) objective and FP8 mixed precision training [5-7].
    *   Supports a 128K context length [6, 7].
    *   Pre-trained on 14.8 trillion tokens [5-7].
    *   Knowledge distillation from DeepSeek-R1 enhances reasoning [5-7].
    *   [Hugging Face Page](DeepSeek-V3 Hugging Face Page)
*   **DeepSeek-R1**
    *   A reasoning-focused model building upon DeepSeek-V3 [8, 9].
    *   Trained with reinforcement learning (RL) to enhance reasoning capabilities [8, 9].
    *   Achieves performance comparable to OpenAI-o1 in maths, code and reasoning [8, 9].
    *   Includes variants like DeepSeek-R1-Zero (trained purely with RL) and DeepSeek-R1 (fine-tuned for clarity) [8, 9].
    *   Supports a 128K context length [9].
    *   Achieves high accuracy on benchmarks like MATH-500 and AIME 2024 [9].
    *   Employs Efficient Supervised Fine-Tuning (ESFT) [2].
    *   [Hugging Face Page](DeepSeek-R1 Hugging Face Page)

#### Training Techniques

*   **Efficient Supervised Fine-Tuning (ESFT):** A training methodology employed by DeepSeek AI to optimise the efficiency and effectiveness of supervised fine-tuning [2]. It aims to maximise performance gains while minimising computational resources and data requirements [2].

### Code Generation Models

#### DeepSeek-Coder

The first generation of code-specialised language models from DeepSeek AI [10, 11]. It assists with code generation, completion, and understanding [10, 11].

*   **Model Versions:** Includes 1.3B, 6.7B, and 33B parameter variants [10, 11].
*   Trained on public code repositories and open-source projects [11].
*   [DeepSeek-Coder 33B Hugging Face Page](DeepSeek-Coder 33B Hugging Face Page)
*   [DeepSeek-Coder 6.7B Hugging Face Page](DeepSeek-Coder 6.7B Hugging Face Page)
*   [DeepSeek-Coder 1.3B Hugging Face Page](DeepSeek-Coder 1.3B Hugging Face Page)

#### DeepSeekCoder-V2

The second generation of code-specialised models, built upon DeepSeek-V2 architecture [12, 13]. It is explicitly designed and optimised for code-related tasks [12, 13].

*   Trained on 2 trillion tokens of code and code-related data [12, 13].
*   Available in Base and Instruct versions [12, 13]. The Instruct version is fine-tuned for following code-related instructions [12, 13].
*   [DeepSeek Coder V2 Hugging Face Page](DeepSeek Coder V2 Hugging Face Page)
*   [DeepSeek Coder V2 Instruct Hugging Face Page](DeepSeek Coder V2 Instruct Hugging Face Page)

### Mathematical Problem-Solving Models

#### DeepSeek-Math

A series of models specialised in mathematical problem-solving, particularly at the competition level [14].

*   **Model Versions:** Includes DeepSeekMath 7B and DeepSeekMath 7B-Instruct [14].
*   Trained on a dataset of mathematical problems, including competition-level problems [14].
*   The Instruct version is fine-tuned for better interaction in mathematical contexts [14].
*   [DeepSeekMath 7B Hugging Face Page](DeepSeekMath 7B Hugging Face Page)
*   [DeepSeekMath 7B Instruct Hugging Face Page](DeepSeekMath 7B Instruct Hugging Face Page)

#### DeepSeek-Prover

Designed for formal theorem proving in Lean 4 [15]. It aims to enhance mathematical reasoning in LLMs, specifically for formal proofs [15].

*   **Model Versions:** Includes DeepSeek-Prover-V1 and DeepSeek-Prover-V1.5 (with Base, SFT, RL, and RL + RMaxTS variants) [15].
*   Trained using a large-scale synthetic dataset of Lean 4 proofs [15].
*   Employs Reinforcement Learning from Proof Assistant Feedback (RLPAF) and RMaxTS for improved proof path exploration [15].
*   [DeepSeek-Prover-V1 Hugging Face Page](DeepSeek-Prover-V1 - Hugging Face)
*   [DeepSeek-Prover-V1.5-RL Hugging Face Page](DeepSeek-Prover-V1.5-RL - Hugging Face)

### Vision-Language Models

#### DeepSeek-VL

The first generation of Vision-Language (VL) models released by DeepSeek AI [16, 17]. It processes and understands both visual and textual information [16, 17].

*   **Model Versions:** Includes Tiny (1.3B), Small (3.9B), and Base (6.7B) variants [16, 17].
*   Trained on image-text pairs and visual question-answering data [17].
*   [Deepseek-ai/deepseek-vl - Hugging Face](Deepseek-ai/deepseek-vl - Hugging Face)
*   [deepseek-ai/deepseek-vl-small - Hugging Face](deepseek-ai/deepseek-vl-small - Hugging Face)

#### DeepSeek-VL2

The second generation of DeepSeek AI's Vision-Language (VL) models [18, 19]. It excels in visual question answering, OCR, document understanding, and visual grounding [18, 19].

*   Built on DeepSeekMoE-27B [18, 19].
*   Employs a Mixture-of-Experts architecture [18, 19].
*   **Model Versions:** Includes Tiny (1.0B activated parameters), Small (2.8B activated parameters), and Base (4.5B activated parameters) variants [18, 19].
*   [Deepseek-ai/deepseek-vl2 - Hugging Face](Deepseek-ai/deepseek-vl2 - Hugging Face)
*   [deepseek-ai/deepseek-vl2-small - Hugging Face](deepseek-ai/deepseek-vl2-small - Hugging Face)

### Mixture-of-Experts (MoE) Models

DeepSeek AI utilises the Mixture-of-Experts (MoE) architecture in several of their models [4, 20]. This approach allows for a large total parameter count while maintaining efficient inference [4, 20]. Only a fraction of the total parameters are activated per token [4]. Key models using MoE include DeepSeek-V2, DeepSeek-V3 and DeepSeek-VL2 [4, 20].

## Important Notes

*   **"DeepSeek-LLM" is a Family Term:** "DeepSeek-LLM" is not a single model but a collective term for DeepSeek AI's text-based Large Language Models [1, 2].
*   **"DeepSeek-V2.5" is Unofficial:** "DeepSeek-V2.5" is not an officially recognised DeepSeek AI model [21, 22]. The term is most likely a community-generated name, informal speculation, or a misunderstanding [22].
*   **Open Access:** DeepSeek AI generally adopts an open-access approach for their base models, releasing model weights on Hugging Face [2, 3, 6-9, 13].

## Contributing

We welcome contributions to this repository! If you have more up-to-date information, corrections, or additional resources related to DeepSeek AI models, feel free to submit a pull request.

## Disclaimer

This repository is a community-driven effort and is not officially endorsed or maintained by DeepSeek AI. All information is based on publicly available resources as of early 2025. For the most accurate and up-to-date details, always refer to the official DeepSeek AI Hugging Face pages and publications.

We hope you find this repository helpful!
Key improvements in this version:
•
Clear Purpose and Structure: The beginning clearly states the repository's purpose and outlines how it is organised.
•
File List: A list of the main files in the repository helps users understand the structure and find specific information.
•
Contributing Section: Encourages community contributions.
•
Disclaimer: Reinforces that this is a community-driven effort and users should refer to official sources for the most up-to-date information.
•
Bolding: Key terms and headings are bolded to improve readability and highlight important information.
•
Citation: Citations have been added to many of the claims to ensure accuracy and verifiability.