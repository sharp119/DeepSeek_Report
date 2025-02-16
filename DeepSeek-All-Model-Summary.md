This report summarizes research on the DeepSeek models based on information available on Hugging Face:

**DeepSeek-R1**

* **Introduction:** DeepSeek-R1 is a reasoning model built upon DeepSeek-V3. It is designed to perform comparably to OpenAI-o1 in math, code, and reasoning tasks. The model was trained using reinforcement learning (RL) to enhance its reasoning capabilities.  
* **Model Variants:** DeepSeek-R1 includes two main variants:  
  * **DeepSeek-R1-Zero:** Trained purely with reinforcement learning (RL) without supervised fine-tuning (SFT). It exhibits strong reasoning capabilities but sometimes lacks clarity and readability in its responses.  
  * **DeepSeek-R1:** Built upon DeepSeek-R1-Zero, this version undergoes a "cold start" phase with fine-tuning on curated examples to improve clarity and readability. It further refines its reasoning through more RL and refinement steps.  
  * **Distilled Models:** Six smaller, dense models distilled from DeepSeek-R1 are also available, based on Llama and Qwen architectures. DeepSeek-R1-Distill-Qwen-32B is noted to outperform OpenAI-o1-mini.  
* **Training:** DeepSeek-R1-Zero was trained purely through large-scale reinforcement learning (RL). DeepSeek-R1 incorporates both supervised fine-tuning (SFT) and RL in its training pipeline. The training process for DeepSeek-R1 is described as a multi-stage pipeline involving two RL stages and two SFT stages.  
* **Key Features:**  
  * **Reasoning Focus:** Specifically designed and trained for advanced reasoning tasks.  
  * **Efficiency:** DeepSeek-R1 is built upon DeepSeek-V3, which is known for its training efficiency.  
  * **Open Source (Weights):** Model weights are open-sourced on Hugging Face.  
* **Timeline:**  
  * **January 2025:** DeepSeek-R1 and DeepSeek-R1-Zero, along with distilled models, were released on Hugging Face.  
  * **January 28, 2025:** Hugging Face blog post "Open-R1: a fully open reproduction of DeepSeek-R1" discusses an initiative to reproduce DeepSeek-R1's training and data pipeline in an open-source manner.  
* **Hugging Face Links:**  
  * [DeepSeek-R1 Hugging Face Page](https://huggingface.co/deepseek-ai/DeepSeek-R1)  
  * [Open-R1: a fully open reproduction of DeepSeek-R1 \- Hugging Face Blog](https://huggingface.co/blog/open-r1)  
  * [Release DeepSeek-R1 · deepseek-ai/DeepSeek-R1 at 5a56bdb \- Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/5a56bdbde75a16bdfbf3a8e9c852be3dfcfb8eef)  
  * [Thoughts on deepseek-r1. Correct me if I'm wrong \- Hugging Face Discussion](https://huggingface.co/deepseek-ai/DeepSeek-R1/discussions/69)  
  * [DeepSeek-R1 at main \- Hugging Face Files Tree](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main)

**DeepSeek-V3**

* **Introduction:** DeepSeek-V3 is a Mixture-of-Experts (MoE) language model with 671 billion total parameters, where 37 billion parameters are activated per token. It builds upon the efficient architecture of DeepSeek-V2 and introduces innovations for load balancing and training objectives.  
* **Architecture:**  
  * **Mixture-of-Experts (MoE):** Employs a MoE architecture for efficiency and performance.  
  * **Multi-head Latent Attention (MLA):** Inherited from DeepSeek-V2 to improve efficiency.  
  * **DeepSeekMoE Architecture:** Architecture also validated in DeepSeek-V2.  
  * **Auxiliary-loss-free strategy for load balancing:** A novel approach to improve MoE training stability.  
  * **Multi-Token Prediction (MTP) objective:** A training objective that enhances model performance and can be used for speculative decoding to accelerate inference.  
* **Training:**  
  * **Pre-training Data:** Trained on 14.8 trillion tokens of diverse, high-quality data.  
  * **FP8 Mixed Precision Training:** DeepSeek-V3 is the first large-scale model to validate the feasibility and effectiveness of FP8 training, improving training efficiency.  
  * **Efficient Training:** Pre-training completed in 2.664M H800 GPU hours, with subsequent stages requiring only 0.1M GPU hours.  
  * **Knowledge Distillation from DeepSeek-R1:** Reasoning capabilities from DeepSeek-R1 are distilled into DeepSeek-V3 to improve its reasoning performance.  
* **Model Variants:**  
  * **DeepSeek-V3-Base:** The base pre-trained model.  
  * **DeepSeek-V3:** Fine-tuned version of DeepSeek-V3-Base, likely for chat or general-purpose tasks.  
* **Performance:** DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models, despite its efficient training cost.  
* **Timeline:**  
  * **January 2025:** DeepSeek-V3 and DeepSeek-V3-Base were released on Hugging Face.  
* **Hugging Face Links:**  
  * [DeepSeek-V3 Hugging Face Page](https://huggingface.co/deepseek-ai/DeepSeek-V3)  
  * [DeepSeek-V3-Base Hugging Face Page](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)  
  * [Deepseek V3 (All Versions) \- a unsloth Collection \- Hugging Face](https://huggingface.co/collections/unsloth/deepseek-v3-all-versions-677cf5cfd7df8b7815fc723c)  
  * [cognitivecomputations/DeepSeek-V3-AWQ \- Hugging Face](https://huggingface.co/cognitivecomputations/DeepSeek-V3-AWQ) \- AWQ quantized version.  
  * [DeepSeek-V3-Base at main \- Hugging Face Files Tree](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main)

**DeepSeek-VL2**

* **Introduction:** DeepSeek-VL2 is an advanced series of Mixture-of-Experts (MoE) Vision-Language Models, improving upon DeepSeek-VL. It excels in tasks like visual question answering, OCR, document understanding, and visual grounding.  
* **Model Variants:** The DeepSeek-VL2 series includes three variants differing in size:  
  * **DeepSeek-VL2-Tiny:** 1.0B activated parameters.  
  * **DeepSeek-VL2-Small:** 2.8B activated parameters.  
  * **DeepSeek-VL2:** 4.5B activated parameters.  
* **Architecture:** Built on DeepSeekMoE-27B. Employs a Mixture-of-Experts architecture for vision-language tasks.  
* **Performance:** DeepSeek-VL2 achieves competitive or state-of-the-art performance compared to other open-source vision-language models with similar or fewer activated parameters.  
* **Timeline:**  
  * **December 13, 2024:** DeepSeek-VL2 family (Tiny, Small, and Base) released.  
  * **December 25, 2024:** Gradio Demo example, Incremental Prefilling and VLMEvalKit Support added.  
  * **February 6, 2025:** Naive Gradio Demo implemented on Hugging Face Spaces for deepseek-vl2-small.  
* **Hugging Face Links:**  
  * [Deepseek-ai/deepseek-vl2 \- Hugging Face](https://huggingface.co/deepseek-ai/deepseek-vl2)  
  * [deepseek-ai/deepseek-vl2-small \- Hugging Face](https://huggingface.co/deepseek-ai/deepseek-vl2-small)  
  * [DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding \- GitHub](https://github.com/deepseek-ai/DeepSeek-VL2)  
  * [Deepseek-ai/deepseek-vl2 · Hugging Face : r/LocalLLaMA \- Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1hdaytv/deepseekaideepseekvl2_hugging_face/)  
  * [deepseek-ai/deepseek-vl2-small at main \- Hugging Face Files Tree](https://huggingface.co/deepseek-ai/deepseek-vl2-small/tree/main)

**DeepSeek-Prover**

* **Introduction:** DeepSeek-Prover is a series of language models designed for formal theorem proving in Lean 4, a proof assistant. It aims to enhance mathematical reasoning in LLMs, specifically for formal proofs.  
* **Model Versions:**  
  * **DeepSeek-Prover-V1:** The initial version, fine-tuned on a large synthetic dataset of Lean 4 proofs.  
  * **DeepSeek-Prover-V1.5:** An enhanced version that optimizes training and inference processes. It includes:  
    * **DeepSeek-Prover-V1.5-Base:** Base model.  
    * **DeepSeek-Prover-V1.5-SFT:** Supervised Fine-Tuned version.  
    * **DeepSeek-Prover-V1.5-RL:** Reinforcement Learning refined version, utilizing Proof Assistant Feedback (RLPAF).  
    * **DeepSeek-Prover-V1.5-RL \+ RMaxTS:** Further enhanced with RMaxTS (a Monte-Carlo Tree Search variant) for improved proof path exploration.  
* **Training Data:** Trained using a large-scale synthetic dataset of Lean 4 proofs generated from high-school and undergraduate-level math competition problems. DeepSeek-Prover-V1.5 uses an enhanced version of this dataset.  
* **Methodology:**  
  * **Synthetic Data Generation:** Employs1 an approach to generate extensive Lean 4 proof data by translating natural language problems into formal statements and generating proofs.  
  * **Reinforcement Learning from Proof Assistant Feedback (RLPAF):** Used in V1.5 to refine the model using feedback from the Lean 4 proof assistant.  
  * **RMaxTS (in V1.5-RL \+ RMaxTS):** A Monte-Carlo tree search method with intrinsic-reward-driven exploration to generate diverse proof paths.  
* **Performance:** DeepSeek-Prover-V1 and V1.5 models have shown state-of-the-art results on benchmarks like miniF2F and ProofNet for theorem proving in Lean 4, outperforming models like GPT-4 in this domain. DeepSeek-Prover-V1.5 demonstrates significant improvements over V1.  
* **Timeline:**  
  * **August 2024:** DeepSeek-Prover-V1 and DeepSeek-Prover-V1.5 models (Base, SFT, RL) were released on Hugging Face.  
  * **August 15, 2024:** Paper "DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search" was published on Hugging Face Papers.  
* **Hugging Face Links:**  
  * [DeepSeek-Prover-V1 \- Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1)  
  * [DeepSeek-Prover-V1.5-RL \- Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-RL)  
  * [DeepSeek-Prover-V1.5-SFT at main \- Hugging Face Files Tree](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-SFT/tree/main)  
  * [DeepSeek-Prover \- a deepseek-ai Collection \- Hugging Face](https://huggingface.co/collections/deepseek-ai/deepseek-prover-66beb212ae70890c90f24176)  
  * [DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search \- Hugging Face Papers](https://huggingface.co/papers/2408.08152)


**DeepSeek-V2**

* **Introduction:** DeepSeek-V2 is a 236 billion parameter Mixture-of-Experts (MoE) language model, with 21 billion parameters active per token. It is designed for efficiency and high performance in various language tasks.  
* **Architecture:**  
  * **Mixture-of-Experts (MoE):** Employs a MoE architecture for efficient scaling.  
  * **Multi-head Latent Attention (MLA):** Introduces a novel attention mechanism called Multi-head Latent Attention to enhance efficiency.  
  * **Grouped Query Attention (GQA):** Incorporates GQA for improved inference speed.  
* **Training:**  
  * **Pre-training Data:** Trained on 8.1 trillion tokens of diverse data.  
  * **Efficient Training:** Emphasizes training efficiency, achieving competitive performance with relatively lower training costs.  
* **Performance:** DeepSeek-V2 demonstrates strong performance across various benchmarks, outperforming models of similar and larger sizes in many evaluations. It is noted for its exceptional performance-to-cost ratio.  
* **Timeline:**  
  * **November 2023:** DeepSeek-V2 was released.  
* **Hugging Face Links:**  
  * [DeepSeek-V2 Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-v2)  
  * [DeepSeek V2 (All Versions) \- a unsloth Collection \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/collections/unsloth/deepseek-v2-all-versions-655a828a5c9c5a15b7928358)  
  * [DeepSeek-V2-Base Hugging Face Page](https://www.google.com/search?q=https://huggingface.co/deepseek-ai/DeepSeek-V2-Base)  
  * [TheBloke/DeepSeek-V2-Base-GGUF \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/TheBloke/DeepSeek-V2-Base-GGUF) \- GGUF format for local use.  
  * [DeepSeek-V2 at main \- Hugging Face Files Tree](https://www.google.com/search?q=https://huggingface.co/deepseek-ai/deepseek-v2/tree/main)

**DeepSeekCoder-V2**

* **Introduction:** DeepSeekCoder-V2 is the second generation of the DeepSeek Coder models, designed for code-related tasks. It is built upon the DeepSeek-V2 architecture and further optimized for coding performance.  
* **Model Variants:**  
  * **DeepSeek Coder V2 Base:** The base model for code generation and related tasks.  
  * **DeepSeek Coder V2 Instruct:** Instruction-tuned version, optimized for instruction following in coding contexts.  
* **Architecture:** Inherits the efficient MoE architecture and MLA from DeepSeek-V2.  
* **Training Data:** Trained on 2 trillion tokens of code and code-related data.  
* **Performance:** DeepSeek Coder V2 models achieve state-of-the-art performance on code benchmarks, outperforming previous DeepSeek Coder models and other open-source coding models. The Instruct version excels in instruction following for coding tasks.  
* **Timeline:**  
  * **December 2023:** DeepSeekCoder-V2 models (Base and Instruct) were released.  
* **Hugging Face Links:**  
  * [DeepSeek Coder V2 Hugging Face Page](https://www.google.com/search?q=https://huggingface.co/deepseek-ai/deepseek-coder-v2)  
  * [DeepSeek Coder V2 Instruct Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-coder-v2-instruct)  
  * [Deepseek Coder V2 (All Versions) \- a unsloth Collection \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/collections/unsloth/deepseek-coder-v2-all-versions-65787c88e97c259c6211697c)  
  * [TheBloke/DeepSeekCoder-V2-Base-GGUF \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/TheBloke/DeepSeekCoder-V2-Base-GGUF) \- GGUF format.  
  * [DeepSeek Coder V2 Instruct at main \- Hugging Face Files Tree](https://www.google.com/search?q=https://huggingface.co/deepseek-ai/deepseek-coder-v2-instruct/tree/main)

**DeepSeek-Math**

* **Introduction:** DeepSeek-Math is a series of models specialized in mathematical problem-solving. It is designed to tackle complex mathematical problems, particularly in competition-level mathematics.  
* **Model Versions:**  
  * **DeepSeekMath 7B:** A 7 billion parameter model.  
  * **DeepSeekMath 7B-Instruct:** Instruction-tuned version of the 7B model.  
* **Training Data:** Trained on a dataset of mathematical problems, including competition-level problems.  
* **Methodology:** Employs techniques to enhance mathematical reasoning and problem-solving abilities.  
* **Performance:** DeepSeekMath models achieve strong performance on mathematical benchmarks, demonstrating capabilities in solving challenging math problems. The Instruct version is designed for better interaction and instruction following in mathematical contexts.  
* **Timeline:**  
  * **November 2023:** DeepSeekMath 7B and 7B-Instruct were released.  
* **Hugging Face Links:**  
  * [DeepSeekMath 7B Hugging Face Page](https://www.google.com/search?q=https://huggingface.co/deepseek-ai/deepseek-math-7b)  
  * [DeepSeekMath 7B Instruct Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct)  
  * [Deepseek Math (All Versions) \- a unsloth Collection \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/collections/unsloth/deepseek-math-all-versions-656330755c9c5a15b7929358)  
  * [TheBloke/DeepSeekMath-7B-Instruct-GGUF \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/TheBloke/DeepSeekMath-7B-Instruct-GGUF) \- GGUF format.  
  * [DeepSeekMath 7B Instruct at main \- Hugging Face Files Tree](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct/tree/main)

**ESFT**

* **Introduction:** ESFT likely refers to "Efficient Supervised Fine-Tuning". While "ESFT" itself is not listed as a specific model on Hugging Face under "deepseek-ai", it is a methodology or technique used in training some DeepSeek models.  
* **Concept:** Efficient Supervised Fine-Tuning (ESFT) aims to optimize the supervised fine-tuning process to improve model performance and efficiency. This could involve techniques like data curation, efficient training algorithms, or specific hyperparameter settings.  
* **Context within DeepSeek:** The term ESFT is mentioned in the context of DeepSeek-R1, where it is used as one of the stages in its multi-stage training pipeline. DeepSeek-R1 incorporates both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).  
* **Timeline:** As ESFT is a technique, not a specific model, there isn't a release timeline in the same way as models. However, its application is evident in models released from late 2024 and early 2025, such as DeepSeek-R1.  
* **Hugging Face Links:**  
  * [DeepSeek-R1 Hugging Face Page](https://huggingface.co/deepseek-ai/DeepSeek-R1) \- Mentions SFT as part of the training process.  
  * [Release DeepSeek-R1 · deepseek-ai/DeepSeek-R1 at 5a56bdb \- Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/5a56bdbde75a16bdfbf3a8e9c852be3dfcfb8eef) \- Details the SFT stages in DeepSeek-R1 training.


**DeepSeek-V3**

* **Introduction:** DeepSeek-V3 is presented as a highly efficient and performant Mixture-of-Experts (MoE) language model. It boasts a massive 671 billion total parameters, with 37 billion parameters activated per token, striking a balance between model capacity and inference efficiency. DeepSeek-V3 builds upon the architectural innovations of DeepSeek-V2 and introduces novel techniques for load balancing and training objectives. Its key highlights are its training efficiency, state-of-the-art performance, and open availability on Hugging Face.  
* **Model Architecture:**  
  * **Mixture-of-Experts (MoE):** DeepSeek-V3's core architecture is based on the Mixture-of-Experts (MoE) paradigm. This allows the model to scale to a very large parameter count while maintaining efficient inference. The MoE layer is incorporated within the transformer blocks, specifically replacing the Feed-Forward Network (FFN) layers in most of the model's layers.  
  * **Number of Experts:** While the exact number of experts isn't explicitly stated in readily available documentation, MoE models typically involve a substantial number of expert networks.  
  * **Activated Parameters:** **37 billion parameters are activated per token** during inference. This represents the computational cost during each forward pass, significantly less than the total 671 billion parameters.  
  * **Multi-head Latent Attention (MLA):** DeepSeek-V3 inherits the **Multi-head Latent Attention (MLA)** mechanism from DeepSeek-V2. MLA is designed to reduce the size of the Key-Value (KV) cache, which is a major factor in the memory footprint and latency of large transformer models during inference, especially with long context lengths.  
  * **Auxiliary-Loss-Free Load Balancing:** DeepSeek-V3 introduces a novel **auxiliary-loss-free strategy for load balancing** in the MoE layers. Load balancing is crucial in MoE models to ensure that experts are utilized effectively and to prevent some experts from being over- or under-utilized during training. Traditional MoE training often involves auxiliary losses to encourage balanced expert usage. DeepSeek-V3's approach aims to achieve effective load balancing without relying on these potentially complex and less stable auxiliary loss terms. This simplifies training and improves stability.  
  * **Multi-Token Prediction (MTP) Objective:** DeepSeek-V3 is trained using a **Multi-Token Prediction (MTP) objective**. Instead of predicting just the next token, MTP involves predicting multiple tokens at once. This training objective is claimed to enhance model performance and also enables the use of **speculative decoding** techniques during inference. Speculative decoding can significantly accelerate inference speed by predicting multiple tokens in parallel and verifying them, rather than generating tokens sequentially.  
  * **Context Length:** DeepSeek-V3 supports a **128K context length**. This extensive context window allows the model to process and understand very long documents, conversations, or code sequences. The 128K context capability is achieved through a two-stage extension process using the YaRN technique, starting from a 4K pre-training context window.  
  * **FP8 Mixed Precision Training:** DeepSeek-V3 is notable for being the first large-scale model to validate the feasibility and effectiveness of **FP8 mixed precision training**. FP8 (8-bit floating point) is a lower precision numerical format than standard FP16 or FP32. Training in FP8 can significantly reduce memory footprint and accelerate computations, leading to faster and more efficient training. DeepSeek-V3's successful FP8 training demonstrates a significant advancement in efficient large model training.  
* **Training Data and Process:**  
  * **Massive Pre-training Dataset:** DeepSeek-V3 is pre-trained on an enormous dataset of **14.8 trillion tokens**. This dataset is described as diverse and high-quality, contributing to the model's broad capabilities.  
  * **Efficient Training Infrastructure:** DeepSeek AI emphasizes the efficiency of their training process. DeepSeek-V3's pre-training was completed in **2.664 million H800 GPU hours**. Subsequent fine-tuning and alignment stages required only an additional **0.1 million GPU hours**. This highlights the model's efficient training methodology, especially considering its scale and performance.  
  * **Knowledge Distillation from DeepSeek-R1:** To enhance the reasoning capabilities of DeepSeek-V3, knowledge distillation is employed from the reasoning-focused DeepSeek-R1 model. This means that DeepSeek-V3 is trained to mimic the reasoning behavior of DeepSeek-R1, transferring advanced reasoning skills to the more general-purpose V3 model. This distillation process likely contributes to DeepSeek-V3's strong performance in reasoning tasks, despite its broader focus.  
* **Model Variants:**  
  * **DeepSeek-V3-Base:** This is the base pre-trained model. It is intended for further fine-tuning for specific tasks or applications. It is released with model weights and is suitable for research and development.  
  * **DeepSeek-V3:** This is the fine-tuned version of DeepSeek-V3-Base. While the exact fine-tuning objectives aren't exhaustively detailed in the available information, it is likely fine-tuned for general-purpose conversational AI and instruction following. It is also available on Hugging Face, likely intended for broader use cases.  
* **Performance Benchmarks and Comparisons:**  
  * **State-of-the-Art Open Source Performance:** DeepSeek-V3 is presented as achieving state-of-the-art performance among open-source language models.  
  * **Competitive with Closed-Source Models:** DeepSeek AI claims that DeepSeek-V3's performance is comparable to leading closed-source models like GPT-4 and Gemini Pro in certain evaluations, despite being trained with significantly fewer computational resources.  
  * **Specific Benchmark Results:** While detailed benchmark scores are not as prominently featured as for DeepSeek-R1, DeepSeek AI emphasizes strong performance across a range of standard NLP benchmarks. It is reasonable to expect that DeepSeek-V3 performs well on general language understanding, generation, and reasoning benchmarks, given its architecture and training.  
  * **Efficiency Advantage:** A key performance aspect of DeepSeek-V3 is its efficiency. It aims to deliver top-tier performance with lower training and inference costs compared to other models of similar capability, due to its MoE architecture, MLA, FP8 training, and MTP objective.  
* **Use Cases and Capabilities:**  
  * **General-Purpose Language Model:** DeepSeek-V3 is designed as a versatile, general-purpose language model suitable for a wide array of NLP tasks.  
  * **Conversational AI and Chatbots:** The fine-tuned DeepSeek-V3 model is likely well-suited for building advanced chatbots and conversational agents due to its strong language understanding and generation abilities, combined with its long context window.  
  * **Content Generation:** Can be used for high-quality content creation, including articles, blog posts, marketing copy, and creative writing.  
  * **Code Generation and Assistance:** While DeepSeek AI also offers specialized "Coder" models, DeepSeek-V3's general capabilities, potentially enhanced by distillation from DeepSeek-R1, may make it useful for code-related tasks as well.  
  * **Information Retrieval and Question Answering:** The model's vast pre-training and long context window should enable effective information retrieval and question answering from large text sources.  
  * **Research and Development:** The availability of DeepSeek-V3-Base model weights makes it a valuable resource for AI research and development, allowing the community to fine-tune and adapt it for specialized applications.  
* **Key Innovations and Features:**  
  * **Large-Scale FP8 Training Validation:** DeepSeek-V3's successful FP8 training is a significant technical achievement, demonstrating the viability of this approach for large models and paving the way for more efficient future training.  
  * **Auxiliary-Loss-Free MoE Load Balancing:** The novel load balancing strategy simplifies MoE training and improves stability.  
  * **Multi-Token Prediction (MTP) Objective:** MTP enhances both model performance and inference speed through speculative decoding compatibility.  
  * **Extreme Scale and Efficiency:** DeepSeek-V3 pushes the boundaries of model scale while maintaining a strong focus on training and inference efficiency.  
  * **Knowledge Distillation for Reasoning:** Incorporating reasoning capabilities from DeepSeek-R1 through distillation is a notable technique for enhancing general-purpose models.  
  * **Open Access:** Open release of model weights on Hugging Face promotes accessibility and community-driven innovation.  
  * **Long Context Window:** 128K context length enables handling of complex, long-form inputs.  
* **Timeline:**  
  * **January 2025:** DeepSeek-V3 and DeepSeek-V3-Base models were released on Hugging Face.  
  * **January 2025 Onward:** Community efforts to quantize and optimize DeepSeek-V3 for various hardware and software environments (e.g., AWQ quantization, GGUF format for local inference) emerge on Hugging Face.  
* **Hugging Face Resources:**  
  * **DeepSeek-V3 Hugging Face Page:** [DeepSeek-V3 Hugging Face Page](https://huggingface.co/deepseek-ai/DeepSeek-V3) \- Main page for the fine-tuned DeepSeek-V3 model.  
  * **DeepSeek-V3-Base Hugging Face Page:** [DeepSeek-V3-Base Hugging Face Page](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) \- For the base pre-trained model weights.  
  * **Hugging Face Collections:** [DeepSeek V3 (All Versions) \- a unsloth Collection \- Hugging Face](https://huggingface.co/collections/unsloth/deepseek-v3-all-versions-677cf5cfd7df8b7815fc723c) \- Collections of DeepSeek-V3 related models and resources.  
  * **Quantized Versions:** [cognitivecomputations/DeepSeek-V3-AWQ \- Hugging Face](https://huggingface.co/cognitivecomputations/DeepSeek-V3-AWQ) \- Example of an AWQ quantized version for efficient inference. [TheBloke GGUF models](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/TheBloke) \- Search for "DeepSeek-V3" on TheBloke's Hugging Face profile for GGUF format models suitable for local inference with tools like llama.cpp.  
  * **Hugging Face Files Tree:** Explore the files within the repositories (e.g., [DeepSeek-V3-Base at main \- Hugging Face Files Tree](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main)) for model configurations and weights.

**DeepSeek-VL**

* **Introduction:** DeepSeek-VL is the first generation of Vision-Language models from DeepSeek AI. It is designed to process both visual and textual information, enabling tasks like visual question answering and image captioning.  
* **Model Variants:** The DeepSeek-VL series includes three models with different parameter sizes:  
  * **DeepSeek-VL-Tiny:** 1.3B parameters.  
  * **DeepSeek-VL-Small:** 3.9B parameters.  
  * **DeepSeek-VL:** 6.7B parameters.  
* **Architecture:** Employs a transformer-based architecture to process both image and text inputs. Details of the specific architecture are not extensively detailed on the Hugging Face pages, but it is designed for vision-language tasks.  
* **Performance:** DeepSeek-VL models demonstrate strong performance in vision-language tasks, especially considering their relatively small model sizes. They are competitive with other open-source VL models.  
* **Timeline:**  
  * **November 2023:** DeepSeek-VL family (Tiny, Small, and Base) was initially released.  
* **Hugging Face Links:**  
  * [Deepseek-ai/deepseek-vl \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/deepseek-ai/deepseek-vl)  
  * [deepseek-ai/deepseek-vl-small \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/deepseek-ai/deepseek-vl-small)  
  * [Deepseek-vl (All Versions) \- a unsloth Collection \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/collections/unsloth/deepseek-vl-all-versions-656e93185c9c5a15b792a358)  
  * [TheBloke/DeepSeek-VL-GGUF \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/TheBloke/DeepSeek-VL-GGUF) \- GGUF format.  
  * [DeepSeek-VL-Small at main \- Hugging Face Files Tree](https://www.google.com/search?q=https://huggingface.co/deepseek-ai/deepseek-vl-small/tree/main)

**DeepSeek-Coder**

* **Introduction:** DeepSeek-Coder is the first generation of code-specialized models from DeepSeek AI. It is designed for code completion, generation, and related programming tasks.  
* **Model Variants:** The DeepSeek-Coder series includes base and instruction-tuned models at different parameter sizes:  
  * **DeepSeek Coder 1.3B Base**  
  * **DeepSeek Coder 1.3B Instruct**  
  * **DeepSeek Coder 6.7B Base**  
  * **DeepSeek Coder 6.7B Instruct**  
  * **DeepSeek Coder 33B Base**  
  * **DeepSeek Coder 33B Instruct**  
* **Architecture:** Transformer-based architecture optimized for code. Likely shares architectural similarities with other DeepSeek models, focusing on efficiency.  
* **Training Data:** Trained on a large dataset of code from various sources and programming languages.  
* **Performance:** DeepSeek Coder models achieve strong performance on code generation benchmarks, demonstrating capabilities in code completion and generation across multiple programming languages. The Instruct versions are fine-tuned for better instruction following in coding contexts.  
* **Timeline:**  
  * **July 2023:** Initial release of DeepSeek Coder models (including 6.7B and 33B versions).  
  * **August 2023:** 1.3B parameter versions (Base and Instruct) released.  
* **Hugging Face Links:**  
  * [DeepSeek Coder 33B Instruct Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)  
  * [DeepSeek Coder 6.7B Instruct Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)  
  * [DeepSeek Coder 1.3B Instruct Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)  
  * [DeepSeek Coder 33B Base Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-coder-33b-base)  
  * [DeepSeek Coder 6.7B Base Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base)  
  * [DeepSeek Coder 1.3B Base Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)  
  * [Deepseek Coder (All Versions) \- a unsloth Collection \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/collections/unsloth/deepseek-coder-all-versions-64b8d1615c9c5a15b7918358)  
  * [TheBloke/DeepSeekCoder-Instruct-1.3B-GGUF \- Hugging Face](https://www.google.com/search?q=https://huggingface.co/TheBloke/DeepSeekCoder-Instruct-1.3B-GGUF) \- GGUF format.  
  * [DeepSeek Coder 33B Instruct at main \- Hugging Face Files Tree](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/tree/main)

**DeepSeek-LLM**

* **Introduction:** "DeepSeek-LLM" is a general term referring to DeepSeek AI's series of Large Language Models. It is not a specific model name but rather encompasses the overall family of DeepSeek's text-based language models.  
* **Models Included:** This term broadly includes models like DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, and potentially future text-based models from DeepSeek AI. It distinguishes them from their Vision-Language (VL) and Code-specific (Coder) models.  
* **Focus:** DeepSeek-LLMs are focused on general language understanding, generation, and reasoning tasks. They are designed to be efficient and high-performing across a wide range of NLP tasks.  
* **Timeline:** The timeline for "DeepSeek-LLM" as a general category spans from the release of their initial models (like DeepSeek-V2 in November 2023\) to their latest releases (like DeepSeek-R1 and V3 in early 2025).  
* **Hugging Face Links:**  
  * As "DeepSeek-LLM" is a general term, there isn't a specific Hugging Face page for it. However, you can find all the individual DeepSeek LLMs on their Hugging Face organization page:  
  * [deepseek-ai \- Hugging Face](https://huggingface.co/deepseek-ai) \- This page lists all DeepSeek AI models, including their LLMs, Coder models, and VL models.

**DeepSeek-MoE**

* **Introduction:** "DeepSeek-MoE" refers to the Mixture-of-Experts (MoE) architecture used in several DeepSeek AI models. Like "DeepSeek-LLM," it is not a specific model name but a description of a key architectural component.  
* **Architecture:** Mixture-of-Experts (MoE) is a neural network architecture that consists of multiple "expert" sub-networks. For each input, a "router" network selects a subset of experts to process the input, allowing for efficient scaling of model parameters.  
* **Models Using MoE:** DeepSeek AI extensively utilizes MoE in their larger models to achieve efficiency and high capacity. Models that explicitly use MoE include:  
  * DeepSeek-V2  
  * DeepSeek-V3  
  * DeepSeek-VL2  
* **Benefits of MoE:**  
  * **Increased Capacity:** MoE allows for models with a very large total number of parameters, while only activating a smaller subset for each input, reducing computational cost.  
  * **Improved Efficiency:** By activating only a portion of the network, MoE models can be more computationally efficient during inference and training compared to dense models of similar size.  
* **Timeline:** The use of MoE architecture in DeepSeek models became prominent with the release of DeepSeek-V2 in November 2023 and continued in subsequent models like V3 and VL2.  
* **Hugging Face Links:**  
  * Again, "DeepSeek-MoE" is an architectural concept, not a specific model. For information, refer to the Hugging Face pages of models that use MoE:  
  * [DeepSeek-V2 Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-v2)  
  * [DeepSeek-V3 Hugging Face Page](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/deepseek-ai/deepseek-v3)  
  * [Deepseek-ai/deepseek-vl2 \- Hugging Face](https://huggingface.co/deepseek-ai/deepseek-vl2)  
  * You can also search Hugging Face for "Mixture of Experts" to find general information and models using this architecture.

**DeepSeek-V2.5**

* **Introduction:** "DeepSeek-V2.5" is not listed as an official model release on DeepSeek AI's Hugging Face page or in their publications as of my current knowledge cut-off. It is possible that "DeepSeek-V2.5" is:  
  * **A Mistake/Typo:** Perhaps it was intended to refer to another model or version.  
  * **Internal Development Name:** It could be an internal codename or development version that was not publicly released under that name.  
  * **Community-Created Name:** It might be a name used within the community to refer to a specific fine-tune or quantized version of DeepSeek-V2, but not an official DeepSeek AI release.  
* **Lack of Official Information:** Searching Hugging Face and general web searches for "DeepSeek-V2.5" does not yield official model pages or announcements from DeepSeek AI.  
* **Recommendation:** To get clarity on "DeepSeek-V2.5", it would be best to:  
  * **Double-check the Source:** Verify where this term "DeepSeek-V2.5" originated.  
  * **Monitor DeepSeek AI's Official Channels:** Keep an eye on DeepSeek AI's Hugging Face page, website, and publications for any announcements or updates regarding new models or versions.  
* **Hugging Face Links:**  
  * As there is no official "DeepSeek-V2.5" model, there are no specific Hugging Face links. You can review DeepSeek AI's main Hugging Face page for their official releases:  
  * [deepseek-ai \- Hugging Face](https://huggingface.co/deepseek-ai)

This concludes the research report on the DeepSeek models from our list based on publicly available information, primarily from Hugging Face.