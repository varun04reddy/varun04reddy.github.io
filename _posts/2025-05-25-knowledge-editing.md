---
title: "Knowledge Editing"
date: 2025-05-25
layout: post
categories: [technical]
---

This post is a more casual commentary on my undergraduate thesis paper, [CaseEdit](https://arxiv.org/abs/2505.19383). While there are certainly things I wish I had implemented or articulated better, I'll set those aside for now and focus on the core idea behind the project: knowledge editing in language models. Working on this topic gave me a foundational understanding of how knowledge is actually represented inside a model's weights, and it helped me begin to make sense of the internal works of LLMs. Through that process, I found myself pulled deeper into the space of mech. interp. My hope is that this read offers a clearer sense of how knowledge is structured within LLMs: how it can be probed, located, and altered.

---

### Background

---

Language models, during the pretraining stage, compress massive amounts of text and images into their parameters in order to learn the statistical patterns that support next-token prediction. Internet, being the largest and most diverse source of text and visual data, it naturally becomes the primary source of pretraining data. However, this only captures a snapshot of the world at a single point in time. While the model is being trained and deployed, the world continues to change, and so does the information landscape. As a result, the model's knowledge is bottlenecked by the timestamp of that initial data snapshot. New facts emerge, and older ones may become outdated or incorrect. One way to address this is by fine-tuning the model on new data, but that approach is resource-intensive and risks forgetting of unrelated knowledge. Knowledge editing offers a more targeted alternative by directly modifying the model's internal memory to update facts.

In modern transformer architectures, factual associations are often embedded within the key and value structures of the model's MLP blocks. These layers, combined with attention mechanisms, allows the model to retrieve and compose knowlege. During inference, attention heads compute a query that interacts with stored keys to retrieve values. These key-value pairs encode patterns of subject and subject-relation representations that map to factual objects. For example, when the model processes a prompt involving a specific subject, the internal activations form a query that retrieves the value associated with the key that matches that subject in context. For a more detailed explanation of keys, queries, and values in attention mechanisms, this [Stack Exchange](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms) post is great.

Knowledge editing methods intervene in this process by identifying and modifying the parameters responsible for producing facts. These methods follow a locate and edit approach. First, they locate the regions of the network that encode a given fact. This involves techniques such as causal tracing or gradient-based attribution to isolate which layers are most influential in producing the original fact. Once identified, a small set of weights (most are found in the MLP layers) is modified to produce a new output for the same query (input). This allows the model's behavior to be updated for specific facts without needing to retrain on large amounts of data or risk overwriting unrelated information.

---

### Factual Knowledge Editing Methods

---

#### MEMIT

[MEMIT](https://arxiv.org/pdf/2210.07229) (Meng, 2023) is a method for performing localized factual edits in transformer models. The core inssight is that facts are often distributied accross various mid-layer MLPs. 

##### Identify Causal Layers

Through causal tracing, MEMIT locates the MLP layers most responsible for factual recall within the language model. In autoregressive models like GPT-J, this often corresponds to a middle block of layers (which is layers 3-8 in GPT-J). MEMIT uses a set of these layers, denoted as $$\mathcal{R}$$, which are used as targets for editing.

##### Distributed Weight Updates

Instead of updating only one layer, MEMIT spreads the update across multiple layers within the selected $$\mathcal{R}$$. For each fact we want to inject (represented as a subject-relation-object tuple $$(s, r, o)$$), MEMIT determines a target hidden vector $$z_i$$ which represents the desired hidden state at the final critical layer $$L$$ (i.e., $$L = \max(\mathcal{R})$$) that would encode the new memory (essentially we are establishing a new target truth).


<figure style="text-align: center;">
  <img src="/assets/img/blog/memit-fig.png" alt="MEMIT Diagram" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1: We see the the "critical path" where MLP layers process factual information about a subject. MEMIT directly edits these MLP modules to store new memories by adjusting the mapping from subject keys to memorized values. (Meng, 2023)</figcaption>
</figure>

The goal is for the final hidden state $$h_i^L$$ to become $$z_i$$. This is done by iteratively modifying the MLP weights in each layer $$l \in \mathcal{R}$$ in ascending order. For each layer $$l$$, it computes the portion of the total required change $$(z_i - h_i^L)$$ that this specific layer should contribute to the residual stream. This portion, denoted as $$r_i^l$$, is distributed as $$r_i^l = \frac{z_i - h_i^L}{L-l+1}$$. This basically means that the deeper layers within $$\mathcal{R}$$ (those with $$l$$ closer to $$L$$) are responsible for a larger remaining portion of the change, while shallower layers contribute smaller fractions of the overall target residual needed.

For each layer $$l$$, MEMIT computes a direct parameter update $$\Delta^l$$ for the MLP's output weights ($$W_{out}^l$$). This update is calculated based on the layer's input activations (keys, $$k_i^l$$) and the target output for that layer ($$m_i^l = W_{out}^l k_i^l + r_i^l$$). The update formula is given by:

$$
\Delta^l = R^l (K^l)^T (C^l + K^l (K^l)^T)^{-1} \quad \text{}
$$

Here, $$K^l$$ is a matrix of input keys for all memories at layer $$l$$, $$R^l$$ is a matrix of the desired residual contributions $$r_i^l$$, and $$C^l$$ is a covariance value holding previously knowledge, which helps balance new vs. old associations. After each layer's update, the model's activations are re-collected to ensure that previous layers process the modified states correctly.


---

#### AlphaEdit

[AlphaEdit](https://arxiv.org/pdf/2410.02355) (Fang, 2025) improves upon MEMIT by introducing null-space projection to reduce the "ripple effect" (knowledge disruptions) during editing. While MEMIT and similar methods attempt to preserve existing knowledge through a regularization term in their objective function, this often creates a trade-off between updating new information and retaining old, potentially leading to model forgetting or collapse (we saw this in the MEMIT update formula earlier).

AlphaEdit, projects the parameter perturbation onto the null space of the preserved knowledge before applying it to the model's weights. This guarantees that the preserved knowledge remains unaffected. This allows AlphaEdit to simplify its objective, focusing solely on minimizing the error of the new, to-be-updated knowledge without compromising existing information. This improvement in editing performance is achieved by adding just a single line of code for this projection (See Figure 2).

<figure style="text-align: center;">
  <img src="/assets/img/blog/alphaedit.png" alt="AlphaEdit Diagram" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 2: AlphaEdit improves upon MEMIT with one line of code. (Fang, 2025)</figcaption>
</figure>



And the results are pretty impressive. This one line of code allows AlphaEdit to significantly reduce the ripple effect, enabling the number of edits to be scaled rapidly. As seen in Figure 3, conventional editing methods, after scaling the number of edits, tend to cause overlaps and convoluted changes, which reduces the quality and accuracy of edits. By solely focusing on updated knowledge via null-space projection, rather than trying to balance the optimization for preserved and updated knowledge, AlphaEdit is minimally affected by a scaled number of edits.

<figure style="text-align: center;">
  <img src="/assets/img/blog/alphaedit-results.png" alt="AlphaEdit Results" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 3: AlphaEdit reduces the ripple effect when scaling edits. (Fang, 2025)</figcaption>
</figure>

---


#### MEMIT-CSK

[MEMIT-CSK](https://arxiv.org/pdf/2305.14956) (Gupta, 2023) is an extension of the original MEMIT editing algorithm, specifically designed to address the nuanced nature of **commonsense knowledge** in LLMs. Unlike normal MEMIT, which was primarily evaluated on factual knowledge with single correct answers, MEMIT-CSK addresses commonsense, characterized by uncertainty with multiple plausible answers. Applying the original MEMIT to commonsense judgments resulted in sub-par performance.

Here is what MEMIT-CSK does to improve on MEMIT for commonsense:
* **Varying Edit Tokens and Positions:** MEMIT focuses on subject token editing], alternatively, MEMIT-CSK allows for editing at various token locations: subject, verb, and object. The core idea is that commonsense plausability depends on each part of the sentence.

* **Improved Layer Selection Strategy:** MEMIT selects a five-layer window for editing whose last layer has the highest AIE (Average Indirect Effect) in the severed causal graph. MEMIT-SK improves this by also considering windows with the maximum *moving average* of AIE, leading to a more robust layer selection. Interestingly, moving AIE showed that commonsense is found in the earlier MLP layers rather than the middle layers (which is where factual knowledge is usally found).

For each fact to be edited, MEMIT-CSK follows a similar pattern to MEMIT by first determining a target hidden vector $$z_i$$ at the final critical layer $$L$$ to encode the new memory. This target is then distributed as a portion $$r_i^l = \frac{z_i - h_i^L}{L-l+1}$$ across the MLP layers in the selected range $$\mathcal{R}$$. The direct parameter update $$\Delta^l$$ for the MLP's output weights ($$W_{out}^l$$) is then computed based on the layer's input activations (keys, $$k_i^l$$) and the target output for that layer ($$m_i^l = W_{out}^l k_i^l + r_i^l$$).


---

### CaseEdit

So how does do these various knowledge edting methods tie together for my thesis? CaseEdit serves as both a dataset creation and an evaluation pipeline specifically designed for commonsense knowledge editing in small-parameter language models. This application is targeted toward personalized, locally hosted home devices, such as edge compute devices that cannot host larger-parameter models. CaseEdit's objective is to test the plausibility of knowledge editing for household LLMs. The pipeline achieves this by generating plausible commonsense household knowledge edits using a larger-parameter model  and producing unique evaluation questions that assess the usefulness of these edits. We then apply flagship knowledge editing techniques to a CaseEdit dataset to compute valuable metrics. **The goal is to evaluate the viablity of household commonsense editing.**

#### Commonsense Edits Generation (CaseEdit)

Our commonsense edits are generated through a multi-stage inference process utilizing a higher-parameter language model (GPT-4o-mini).

* **Step 1: Generate Atypical Location.** First, for each selected household object (subject), an independent inference loop is used to propose an unusual, yet plausible, everyday household location. For example, while a butter knife is typically found in the kitchen, our language model might suggest an unusual location like a garage.

* **Step 2: Generate New Ground Truth.** In a subsequent inference loop, we provide the household item and the generated unusual location. The model is then prompted to generate an atypical use or property for that item, conditioned on a randomly assigned "Plausibility Bucket" (See [ATOMIC 2020](https://arxiv.org/pdf/2010.05953) for more about this). This establishes a new target truth for the edit (See Table 1).

<figure style="text-align: center;">
  <img src="/assets/img/blog/caseedit-table.png" alt="Caseedit Table1" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Examples of CaseEdit knowledge editing chain creation pipeline. (Reddy, 2025)</figcaption>
</figure>


For the **Evaluation Question Generation**, additional inference loop is employed to create four types of multiple-choice questions for each edit. These questions are tailored to evaluate the knowledge edits across key metrics (See Table 2):
* **Reliability:** Directly assesses if the edit successfully modifies the model's output for the specific input.
* **Generalization:** Evaluates if the model correctly applies the edited knowledge to semantically similar inputs or paraphrased versions.
* **Locality:** Determines if the edit unintentionally alters predictions on unrelated inputs, assessing for negative "ripple effects".
* **Portability:** Measures if the newly acquired knowledge can be correctly applied in more complex, multi-hop reasoning scenarios or downstream tasks.

<figure style="text-align: center;">
  <img src="/assets/img/blog/caseedit-table2.png" alt="CaseEdit Table 2" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Examples of CaseEdit evaluation questions. Tokens activating the edited layer are highlighted in blue, while potentially entangled tokens that should remain unchanged are highlighted in red.(Reddy, 2025)</figcaption>
</figure>

We posit that larger parameter models inherently possess a more robust understanding of commonsense. This distinction forms the bedrock of our approach for two key reasons:

1.  **Enabling Personalized Small-Parameter Assistants:** Smaller parameter models are ideal for edge computing environments due to their lightweight architecture, real-time adaptability, and energy efficiency. These models inherently face challenges in adapting to highly personalized or context-specific commonsense requirements. We therefore advocate for knowledge editing as an effective method to imbue these smaller models with the necessary, context-driven commonsense, allowing them to function as more effective and personalized assistants.

2.  **Leveraging Large Models for Ground Truth Generation:** Our confidence in using a larger parameter model (specifically, GPT-4o-mini) for generating new commonsense ground truths stems from their demonstrated superior capabilities in reasoning and adapting to nuanced contexts. This allows us to curate high-quality, atypical household-specific knowledge, which then serves as the target for editing in smaller models.

To illustrate the difference in inherent commonsense capabilities, beyond the scope of direct editing performance, we conducted an evaluation using a generated CaseEdit dataset. This involved assessing a small-parameter Llama-3.1-8B model against a larger-parameter Llama-3.1-70B model. In this setup, we used an altered version of our multiple-choice questions, removing the original ground truth and replacing it with a throwaway option. For each answer the models choose, we prompt them to provide an explanation. The alignment of these explanations with the chosen answer is assessed, ideally by a human evaluator, but in our case a larger parameter: Llama 3.1 405B.

<table style="width:100%; border-collapse:collapse;">
  <tr>
    <th style="padding: 8px;">Model</th>
    <th style="padding: 8px;">Reliability</th>
    <th style="padding: 8px;">Generalization</th>
    <th style="padding: 8px;">Locality</th>
    <th style="padding: 8px;">Portability</th>
  </tr>
  <tr>
    <td style="padding: 8px;">Llama-3.1-8B</td>
    <td style="padding: 8px;">65%</td>
    <td style="padding: 8px;">61%</td>
    <td style="padding: 8px;">54%</td>
    <td style="padding: 8px;">39%</td>
  </tr>
  <tr>
    <td style="padding: 8px;">Llama-3.1-70B</td>
    <td style="padding: 8px;">84%</td>
    <td style="padding: 8px;">81%</td>
    <td style="padding: 8px;">82%</td>
    <td style="padding: 8px;">79%</td>
  </tr>
</table>


---

Our experiments, particularly with AlphaEdit, yielded impressive results. AlphaEdit consistently outperformed other KE methods on CaseEdit across all evaluated metrics, demonstrating its strong performance even in personalized commonsense KE. Additonally, as we scaled the number of edits, AlphaEdit showed minimal ripple effects.

We also analyzed the model's confidence and uncertainty as edits scaled. This was done by examining the softmax probability distribution over the five multiple-choice answers, derived from the model's output logits. To quantify this uncertainty, we employed Shannon entropy $$H(p)$$ of the probability distribution:

$$H(p) = -\sum_{i \in \{A,B,C,D,E\}} p_i \log_2(p_i) \quad $$

As seen in Figure 4: 

<figure style="text-align: center;">
  <img src="/assets/img/blog/caseedit-logit.png" alt="CaseEdit Table 2" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Next-token probabilities and entropy during MCQ evaluation using AlphaEdit on CaseEdit.</figcaption>
</figure>

When we see the model initially placing a high probability on the original, pre-edit truth, it tells us that this piece of information is very strongly and consistently encoded within the LLM's vast set of parameters. It's a clear signal that the model's internal knowledge representation firmly aligns with that fact.

Now, after we've applied a single edit, and the probability mass has largely shifted to our new ground truth, this signifies a successful intervention. The editing method has effectively rerouted the model's internal associations, so that when queried, it now confidently points to the updated information. However, the presence of residual probability on the old truth or other options indicates that the edit isn't always a complete overwrite; it might introduce a slight degree of ambiguity or retain faint, underlying connections to the prior information. 

The real technical challenge, and what Figure 4 really highlights, comes with sequential editing. The gradual decrease in probability for a newly edited fact, combined with an increase in overall uncertainty, shows us the "ripple effect" in action at a fundamental level. It means that as we modify more and more parts of the model's knowledge, even seemingly unrelated edits can subtly interfere with previously updated facts, causing the model's internal confidence in those earlier edits to dilute.

This is where AlphaEdit shines: by projecting parameter perturbations into the null space of preserved knowledge, it minimizes cumulative interference. It means the model's internal representation of each individual edited fact is more isolated and therefore less prone to "forgetting" or "overwriting" when subsequent knowledge updates occur.

---

### Conclusion

In this post, we walked through the landscape of knowledge editing in language models. Along the way, we introduced CaseEdit—a dataset and evaluation pipeline built for probing commonsense edits in small-parameter LLMs. We also explored how techniques like MEMIT and AlphaEdit interact with a model’s internal memory, each bringing its own approach to the difficult task of rewriting what a model "knows."

That said, there is still a long road ahead. Many directions remain untapped, partly due to the usual suspects—limited compute and time—but also because this space is still very much under active construction.

**RAG vs. Weight-Based Editing**  
An open question that keeps coming up is how weight-based knowledge editing compares to retrieval-based approaches like RAG. Both promise to update a model’s behavior, but with very different philosophies. One rewires the model’s internals directly, while the other fetches knowledge from an external source at inference time. A compelling next step would be to evaluate both paradigms under CaseEdit’s contextual edit tasks, especially those that involve subtle multi-hop commonsense. My guess is that each will have blind spots that the other can help cover.

**Blending Mechanisms for a Hybrid Approach**  
I am also curious about what happens when we mix the strengths of MEMIT-CSK and AlphaEdit. MEMIT-CSK stands out for its flexible editing across token types and smarter layer selection. AlphaEdit brings a crisp linear algebra perspective with its null-space projection strategy. It seems natural to ask whether AlphaEdit’s minimal injection style could be merged with MEMIT-CSK’s more distributed edit process. A hybrid method might capture the best of both worlds—structured propagation with tight generalization control.

**Tracing Internal Model Circuits:**  
Lastly, there is the interpretability angle. We still do not fully understand what happens inside the model after an edit is made. What changes, exactly? Which pathways reconfigure themselves? Future work could trace the circuits involved before and after an edit, mapping how the model’s "belief network" morphs to accommodate a new fact. This might be one of the most promising ways to turn black-box interventions into something more surgical and predictable.

Working on knowledge editing gave me a much deeper understanding of how large language models store and access information. I believe anyone interested in the internals of these systems can benefit from exploring this area, even briefly. By learning how factual edits propagate through a model's layers, I began to form a more concrete intuition about where information resides—how it is encoded across MLP layers, how it interacts with attention, and how specific facts emerge from distributed representations. It is not interpretable in the way humans might prefer, but it is consistent and surprisingly structured once you know where to look.

That said, I have come to believe that knowledge editing is unlikely to be the long-term solution for keeping models up to date. There are two main reasons. First, efficiency. Even while working with a relatively small model, I ran into bottlenecks related to compute and time. Making hundreds of targeted edits, validating each one, and managing interference is far from lightweight. Second, effectiveness. While editing methods like MEMIT and AlphaEdit work well in many cases, they are not universally reliable. They can struggle with generalization, or with preserving fluency in edge cases.

This is why I suspect that state-of-the-art systems will continue leaning toward tool calling with search, rather than purely parameter-based updates. Tooling with search provides fresher knowledge at inference time, with minimal risk to model coherence or memory. Still, knowledge editing has value. If nothing else, it offers a lens into the the black boxiness of LLMs which can help inform future work on model interpretability and internal representation design. 











#### References

---

* Meng, K., Sen Sharma, A., Andonian, A. J., Belinkov, Y., & Bau, D. (2023). *Mass-editing memory in a transformer*. ICLR 2023.

* Fang, J., Jiang, H., Wang, K., Ma, Y., Shi, J., Wang, X., He, X., & Chua, T.-S. (2025). *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models*. ICLR 2025.

* Reddy, V., Kuo, Y. (2025). *CaseEdit: Enhancing Localized Commonsense Reasoning via Null-Space Constrained Knowledge Editing in Small Parameter Language Models*. UVA SEAS Library.

* Gupta, A., Mondal, D., Sheshadri, A. K., Zhao, W., Li, X. L., Wiegreffe, S., & Tandon, N. (2023). *Editing Common Sense in Transformers*. EMNLP 2023.

* Jena D. Hwang and Chandra Bhagavatula and Ronan {Le Bras} and Jeff Da and Keisuke Sakaguchi and Antoine Bosselut and Yejin Choi. (2021). *(COMET)ATOMIC2020:On Symbolic and Neural Commonsense Knowledge Graphs*. AAAI 2021.

