---
title: "Mathematical Foundations for RAG Systems"
category: "GenAI Knowledge"
date: "2024-01-26"
summary: "Exploring the mathematical concepts behind Retrieval-Augmented Generation systems"
tags: ["Mathematics", "RAG", "Vector Search", "Embeddings", "Attention"]
author: "Haoyang Han"
---

# Mathematical Foundations for RAG Systems

Understanding the mathematical foundations behind Retrieval-Augmented Generation (RAG) systems is crucial for building effective implementations. This post explores the key mathematical concepts with beautiful equation rendering.

## Vector Embeddings and Similarity

### Cosine Similarity

The fundamental measure for document similarity in RAG systems is cosine similarity. For two vectors $\mathbf{a}$ and $\mathbf{b}$, the cosine similarity is defined as:

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| |\mathbf{b}|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$

This formula ranges from -1 to 1, where 1 indicates perfect similarity, 0 indicates orthogonality, and -1 indicates opposite directions.

### Euclidean Distance

An alternative similarity measure is the Euclidean distance between vectors:

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

The similarity can then be computed as $\text{similarity} = \frac{1}{1 + d(\mathbf{a}, \mathbf{b})}$.

## Attention Mechanisms

### Self-Attention

The self-attention mechanism used in transformers computes attention weights as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ is the query matrix
- $K$ is the key matrix  
- $V$ is the value matrix
- $d_k$ is the dimension of the key vectors

### Multi-Head Attention

Multi-head attention extends this concept by computing multiple attention heads in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Where each head is computed as:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## Probability and Information Theory

### Cross-Entropy Loss

The training objective for language models often uses cross-entropy loss:

$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{j=1}^{V} y_{i,j} \log(\hat{y}_{i,j})$$

Where $y_{i,j}$ is the true probability and $\hat{y}_{i,j}$ is the predicted probability for token $j$ at position $i$.

### Perplexity

Model performance is often measured using perplexity, which is the exponentiated cross-entropy:

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1})\right)$$

Lower perplexity indicates better model performance.

## Retrieval Scoring

### TF-IDF Scoring

Traditional information retrieval uses TF-IDF (Term Frequency-Inverse Document Frequency):

$$\text{TF-IDF}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$

Where:
- $\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$ is the term frequency
- $\text{idf}(t, D) = \log\frac{|D|}{|\{d \in D : t \in d\}|}$ is the inverse document frequency

### BM25 Scoring

BM25 (Best Matching 25) is an advanced ranking function:

$$\text{BM25}(q, d) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

Where:
- $f(q_i, d)$ is the frequency of query term $q_i$ in document $d$
- $|d|$ is the length of document $d$
- $\text{avgdl}$ is the average document length
- $k_1$ and $b$ are tuning parameters

## Vector Quantization

### Product Quantization

For efficient storage, vectors can be quantized using Product Quantization:

$$\mathbf{x} = [\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_m]$$

Each subvector $\mathbf{u}_j$ is quantized to the nearest centroid:

$$q(\mathbf{u}_j) = \arg\min_{c \in C_j} ||\mathbf{u}_j - c||^2$$

The quantized vector becomes:

$$\hat{\mathbf{x}} = [q(\mathbf{u}_1), q(\mathbf{u}_2), \ldots, q(\mathbf{u}_m)]$$

## Optimization and Learning

### Gradient Descent

The fundamental optimization algorithm for training neural networks:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$$

Where $\alpha$ is the learning rate and $\nabla_\theta \mathcal{L}$ is the gradient of the loss function.

### Adam Optimizer

Adam combines momentum and adaptive learning rates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

The parameter update becomes:

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected estimates.

## Evaluation Metrics

### Mean Reciprocal Rank (MRR)

For retrieval evaluation:

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the position of the first relevant document for query $i$.

### Normalized Discounted Cumulative Gain (NDCG)

NDCG@k measures ranking quality with position-based discounting:

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

Where DCG@k is:

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

## Practical Implementation

### Efficient Vector Search

For large-scale systems, approximate nearest neighbor search uses techniques like LSH (Locality-Sensitive Hashing):

$$h(\mathbf{v}) = \text{sign}(\mathbf{a} \cdot \mathbf{v} + b)$$

Where $\mathbf{a}$ is a random vector and $b$ is a random scalar.

### Memory Efficiency

The memory complexity for storing $N$ vectors of dimension $d$ is $O(Nd)$. With quantization, this reduces to:

$$\text{Memory} = N \times \left(\frac{d}{m} \times \log_2(k) + m \times k \times \frac{d}{m}\right)$$

Where $m$ is the number of subvectors and $k$ is the number of centroids per subvector.

## Conclusion

These mathematical foundations form the backbone of modern RAG systems. Understanding concepts like attention mechanisms (with scaling factor $\frac{1}{\sqrt{d_k}}$), similarity measures, and optimization algorithms enables building more effective retrieval-augmented systems.

The interplay between these mathematical concepts allows RAG systems to:
1. Encode semantic meaning through vector embeddings
2. Efficiently retrieve relevant information
3. Generate contextually appropriate responses
4. Optimize performance through proper evaluation metrics

Mastering these mathematical principles is essential for advancing the field of retrieval-augmented generation. 