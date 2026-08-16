# PragmaGap

**Quantifying the Surface-to-Deep Representation Gap in Arabic Transformers Using Qur'anic Metaphors**

PragmaGap is a research project that investigates whether Arabic Transformer encoders represent **surface grammatical structure** and **deeper pragmatic intent** with comparable effectiveness.

The project focuses on Qur'anic metaphors as linguistically rich expressions in which grammatical form and communicative function may not always provide the same level of information.

## Research Motivation

A central question in computational linguistics is whether pretrained language models capture linguistic information beyond observable surface patterns.

For figurative language, this question is particularly important. A metaphor may exhibit identifiable **grammatical and structural properties**, while its interpretation also depends on its **communicative function and pragmatic intent**.

PragmaGap therefore investigates the following question:

> **Do Arabic Transformer encoders perform equally well when probing surface grammatical structure and deeper pragmatic function in Qur'anic metaphors?**

The project operationalizes this question as a **Surface-to-Deep Representation Gap**.

## Linguistic Framework

The study compares two levels of linguistic representation:

- **Surface / Structural Level**  
  Grammatical structure associated with the metaphorical expression.

- **Deep / Pragmatic Level**  
  The communicative function represented through speech-act categories.

To obtain sufficiently dense categories for a relatively small dataset, the original labels are linguistically grouped into two broader dimensions:

### Grammatical Structure

- **Verbal / Dynamic**
- **Nominal / Static**

### Speech Act

- **Assertive / Informative**
- **Affective / Action-Oriented**

The grouping is intended to preserve a meaningful linguistic distinction while reducing sparsity caused by highly fine-grained categories.

## The PragmaGap Hypothesis

The central hypothesis is that Transformer representations may encode **grammatical regularities more reliably than pragmatic functions**.

This is expressed through the performance difference:

\[
\Delta = F1_{\text{Syntax}} - F1_{\text{Pragmatics}}
\]

A larger positive value of \( \Delta \) indicates a stronger gap between structural and pragmatic probing performance.

The comparison is conducted across multiple Arabic pretrained Transformer encoders to examine whether this gap varies according to their pretraining linguistic environment.

## Models

The study evaluates five pretrained Transformer encoders:

- AraBERTv2
- ARBERT
- CAMeLBERT-CA
- MARBERTv2
- XLM-R

The models provide different pretraining backgrounds, allowing the study to examine how linguistic domain and training data may influence structural and pragmatic representations.

## Dataset

The experiments use approximately **950 Qur'anic metaphorical examples** annotated with linguistic and pragmatic information.

The dataset is treated as a low-resource setting. Rather than relying on a single train/test split, the experiments employ **5-fold cross-validation** to obtain more stable estimates across the available examples.

## Experimental Perspective

The project is framed as a **dual probing study** rather than a conventional metaphor-classification task.

The primary comparison is therefore not simply:

> Which model achieves the highest score?

Instead, the study asks:

> **What type of linguistic information is more readily represented by each model: grammatical structure or pragmatic function?**

This distinction makes the model comparison linguistically interpretable.

## Evaluation

The primary evaluation metric is **Macro-F1**, which is appropriate for comparing performance across potentially imbalanced categories.

Results are examined across folds and models to evaluate:

- Structural probing performance
- Pragmatic probing performance
- The Surface-to-Deep Gap (\(\Delta\))
- Stability of the observed differences
- Statistical significance of the gap

## Research Scope

PragmaGap is intended as a contribution to research at the intersection of:

- Computational Linguistics
- Arabic NLP
- Pragmatics
- Computational Rhetoric
- Figurative Language Processing
- Linguistic Probing
- Transformer Representation Analysis

The project does **not** claim that successful grammatical probing constitutes linguistic understanding. Instead, it uses the contrast between structural and pragmatic probing as an empirical method for examining the distribution of linguistic information encoded by pretrained representations.
