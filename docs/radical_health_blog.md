# radical health blog on medical embeddings

Our new medical EmbeddingGemma-300m fine-tuned on medical data can predict diseases from patient notes with an AUROC of 0.934 and provide increased survival analysis!

## The Problem with Embedding Models
The current problem with embedding models is that they are trained to match text content, but text covers a whole range of information, from language to style, to semantics, to the underlying information or instructions. When it comes to needing embedding which contain the specific information and are agnostic to stylistic information, off-the-shelf embeddings typically fail.

For the medical context this is critical, as we need to be able to faithfully extract the correct clinical information and prioritise this over other stylistic information.

## Fine-tuning our own Embedding Model for clinical utility
We built the first generation of an internal embedding model by fine-tuning the EmbeddingGemma-300m using the MIMIC-III dataset, which is a large, freely available critical care database containing de-identified health records from over 40,000 ICU patients between 2001 and 2012. Using the contrastive loss, we fine-tune the embedding model by setting the anchor to be a patient note at time t and the positive to be a patient note at t + 1. This forces the model to match notes based on the medical context and not to rely on the style of writing, which is typically consistent between notes.

Fine-tuning a model this way yields surprisingly powerful results. Given a recall task of recalling the next patient note, our model is able to achieve a top5 accuracy of 65%, far surpassing the accuracy of the base model and OpenAI embeddings, which achieve 6% and 9% respectively.

This isn’t surprising as we’re optimising this in the loss, but where this gets exciting is that these representations make for much better medical performance on downstream tasks, our model beats the base model and OpenAI when trying to predict diagnosis: ours: 0.934; openAI: 0.809; base: 0.674 (AUROC), and when performing survival analysis ours 0.70; OpenAI 0.67; base 0.59 (C-Index). 

We can clearly see why the model is so powerful at producing these results when we create UMAP plots and color code by common disease types. 