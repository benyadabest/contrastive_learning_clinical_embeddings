# A Model That Understands Your Medical Notes

**Benjamin Shvartsman (bs828), Timothy Lin (tql4), Gaspard Loeillot (gl458)**

**Keywords:** natural language processing, clinical text, long-context representation learning, contrastive learning, embedding fine-tuning, EHR

**Application Setting:** The model powers patient- and staff-facing copilots operating over longitudinal health records, supporting one (or more) tasks from: intake note understanding, prior authorization packet generation, and/or care navigation/onboarding workflows for complex care.

---

## Big Picture & What We Want to Do

General-purpose embedding models from providers like OpenAI encode text along many dimensions (style, syntax, topic), but they don't prioritize the clinical semantics that are critical in healthcare. When two discharge summaries describe the same patient trajectory in different prose styles, a general embedding model may push them apart based on surface-level differences rather than pulling them together based on shared medical context. This gap is consequential: downstream tasks like disease prediction, risk stratification, and survival analysis depend on embeddings that faithfully capture clinical state rather than writing conventions. Radical Health AI recently demonstrated this principle by fine-tuning EmbeddingGemma-300m on MIMIC-III patient notes using a temporal contrastive objective, achieving a 0.934 AUROC on diagnosis prediction versus 0.809 for OpenAI. We aim to reproduce, validate, and extend that result.

Our project will replicate Radical Health's contrastive fine-tuning pipeline as a baseline, then systematically improve it along three axes:

1. **Richer training signal** through hierarchical contrastive objectives that incorporate structured clinical labels (ICD codes, procedure codes, medication classes) alongside the temporal note-pair signal.
2. **Deployment** as a production-ready embedding service on Baseten, integrated with Fasten Health's patient record aggregation layer to demonstrate real-world utility.

The result will be an end-to-end system from raw clinical text to actionable clinical intelligence.

---

## Core AI Approach: Contrastive Learning for Clinical Embeddings

The foundational technique is contrastive representation learning applied to sequential clinical notes. Following Radical Health's approach, we set the anchor as a patient note at time *t* and the positive as the same patient's note at time *t+1*, with in-batch negatives drawn from other patients. The contrastive loss (InfoNCE) forces the model to learn representations that capture the latent medical trajectory of a patient — their evolving conditions, treatments, and outcomes — while becoming invariant to stylistic differences between clinicians, departments, or documentation templates. We will use EmbeddingGemma-300m as our base model, a compact (300M parameter) model from Google designed for efficient embedding generation, which is small enough to fine-tune on accessible GPU budgets via Baseten's training infrastructure.

To create richer training signal incorporating structured clinical labels, we will, for a given note at time *t*, generate embeddings for those labels corresponding with the same patient at or near *t*. Here, we intend to use hierarchical contrasting objectives rather than regular contrasting learning to leverage the hierarchical nature of these structured clinical labels. This would allow us to generate a set of embeddings that differ from each other depending on distance in the hierarchy rather than bluntly penalizing nonmatches (e.g., embeddings for ICD codes in the same category might differ less than those for ICD codes in different categories but the same chapter). To this end, we are considering using **HiMulCon** or a similar alternative as our loss function. We will augment our embeddings for notes with these embeddings for the structured labels.

---

## Evaluation

As we intend to replicate the Radical Health contrastive fine-tuning pipeline as a baseline, our evaluation approach will be similar.

**Note Recall:** We will assess how often the embedding of a note for a patient at time *t+1* is similar to the embedding of a note for that patient at time *t*, determining whether our embeddings truly capture patient trajectory information. Target: **top-5 accuracy of 65%**, in line with the Radical Health baseline.

**Diagnosis Prediction:** We will assess accuracy by using embeddings as input features for downstream clinical prediction tasks. Diagnosis prediction will be framed as a multi-label classification problem using ICD-9 diagnosis codes from MIMIC-III as ground truth labels. We will train lightweight classifiers (logistic regression, gradient-boosted trees) on frozen embeddings and report **AUROC** and **top-k accuracy** metrics.

**Model Comparison:** We will compare four embedding variants:

| Model | Type |
|---|---|
| OpenAI text-embedding-3-small | General-purpose baseline |
| OpenAI text-embedding-3-large | General-purpose baseline |
| EmbeddingGemma (temporal contrastive) | Fine-tuned baseline |
| EmbeddingGemma (hierarchical contrastive) | Our proposed extension |

All models will be evaluated on identical train/validation/test splits. Success is defined as consistent and statistically meaningful improvements over both OpenAI baselines and the temporal-only baseline. We will also perform **UMAP** dimensionality reduction for qualitative analysis of clustering patterns across diagnoses and patient trajectories.

---

## Timeline

| Week | Milestone & Tasks |
|---|---|
| **Feb 16** | Finalize experimental design. Define train/validation/test splits for MIMIC-III. Set up reproducible preprocessing pipeline (note extraction, patient grouping, temporal ordering). |
| **Feb 23** | Implement baseline embedding inference using OpenAI embeddings and EmbeddingGemma (no fine-tuning). Run initial note-recall evaluation to establish baseline metrics. |
| **March 2** | Implement temporal contrastive fine-tuning pipeline (InfoNCE loss, in-batch negatives). Verify training stability on small subset. |
| **March 9** | Full-scale temporal contrastive training on MIMIC subset. Monitor convergence. Conduct initial note-recall evaluation. Debug instability or collapse if needed. |
| **March 16** | Complete temporal baseline training. Run full note-recall benchmark. Begin downstream diagnosis prediction experiments using frozen embeddings. |
| **March 23** | Analyze baseline results. Identify weaknesses. Begin implementation of hierarchical contrastive objective incorporating ICD structure. |
| **March 30** | Train hierarchical contrastive variant on reduced dataset. Validate loss behavior and gradient stability. Run small-scale evaluation. |
| **April 6** | Full hierarchical training run. Generate embeddings and evaluate on note recall + diagnosis prediction. Compare against temporal-only baseline. |
| **April 13** | Conduct ablation studies (loss weighting, batch size, dimensionality if feasible). Perform UMAP visualizations and embedding geometry analysis. |
| **April 20** | Iterate based on findings. Adjust hyperparameters if performance gaps observed. Final training run if needed. Freeze final model. |
| **April 27** | Integrate embedding model into Baseten deployment pipeline. Demonstrate inference through the Fasten Health integration layer. Prepare demo artifacts. |
| **Early May** | Write final report: methodology, experiments, results, ablations, analysis. Prepare figures, tables, and demo presentation. Buffer time for unexpected issues. |

---

## Existing Resources

### Software

1. **Baseten** — Cloud MLOps platform for model fine-tuning and inference. Provides GPU-backed training jobs, optimized embedding inference via BEI (Baseten Embeddings Inference), REST API deployment, and autoscaling. HIPAA-compliant and SOC 2 Type II certified. We will use Baseten for all training and deployment.

2. **Fasten Health** — Open-source personal health record aggregator integrating with 50,000+ US healthcare systems via FHIR APIs. We will use Fasten Health's data model and API layer as the integration surface for our embedding service, demonstrating how fine-tuned embeddings can power applications on real patient-authorized records.

3. **Hugging Face Transformers / Sentence-Transformers** — For loading EmbeddingGemma-300m, implementing contrastive training loops, and managing model checkpoints.

4. **scikit-learn, XGBoost, lifelines** — For downstream evaluation classifiers (logistic regression, gradient-boosted trees) and survival analysis (Cox proportional hazards via lifelines library).

5. **UMAP, matplotlib** — For embedding space visualization and qualitative evaluation.

### Data

1. **MIMIC-III Clinical Database** (available via Kaggle) — De-identified health records from 40,000+ ICU patients at Beth Israel Deaconess Medical Center (2001–2012). Key tables: `NOTEEVENTS` (2M+ clinical notes), `ADMISSIONS`, `DIAGNOSES_ICD`, `PROCEDURES_ICD`, and `PATIENTS`. This is the same dataset used by Radical Health, enabling direct comparison. Data is already downloaded and verified.

2. **OpenAI Embedding API** — `text-embedding-3-small` and `text-embedding-3-large` as comparison baselines. API access is available and budgeted for evaluation-only usage.

---

## References

1. Radical Health AI. "Training a model that understands your notes 7x better than OpenAI." Blog post, April 2025. https://radicalhealth.ai/blog/training-a-model-that-understands-your-notes-7x-better-than-openai

2. Gao, T., Yao, X., & Chen, D. "SimCSE: Simple contrastive learning of sentence embeddings." *EMNLP 2021.*

3. Oord, A. v. d., Li, Y., & Vinyals, O. "Representation learning with contrastive predictive coding." *arXiv:1807.03748* (2018).

4. Baseten. "High-performance embedding model inference." https://www.baseten.co/resources/guide/high-performance-embedding-model-inference/

5. Fasten Health. Open-source personal health record aggregator. https://github.com/fastenhealth/fasten-onprem

6. Zhang, S., Xu, R., Xiong, C., & Ramaiah, C. "Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework." *arXiv:2204.13207* (2022).