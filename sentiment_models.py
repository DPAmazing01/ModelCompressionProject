"""
sentiment_models.py
-------------------
Three models for binary sentiment classification on the IMDB dataset.

  TextCNN  - Multi-filter convolutional network over word embeddings (Kim 2014)
  BERT     - Pretrained BERT encoder with classification head via HuggingFace
  BiLSTM   - Bidirectional LSTM with final hidden state classification

Input conventions
  TextCNN / BiLSTM : (batch, seq_len)  token ID sequences from a shared vocabulary
  BERT             : expects HuggingFace tokenizer outputs:
                       input_ids      (batch, seq_len)
                       attention_mask (batch, seq_len)
                     passed as keyword arguments (see build_bert usage note)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# TextCNN - Multi-Filter Convolutional Text Classifier
# =============================================================================
# Implements the architecture from Kim (2014) "Convolutional Neural Networks
# for Sentence Classification".  Multiple parallel Conv2d filters of different
# heights scan over the embedding matrix; max-over-time pooling collapses each
# filter's output to a single value; the concatenated pool vector is classified.
#
# The Conv2d formulation (with a single input channel of height seq_len x
# embed_dim) is equivalent to the original 1D convolution formulation but
# is more commonly seen in PyTorch implementations.
# =============================================================================

class TextCNN(nn.Module):
    """
    Multi-filter CNN for text classification.

    Args:
        vocab_size   : vocabulary size (set from dataset tokenizer)
        embed_dim    : word embedding dimensionality
        num_classes  : number of output classes (2 for IMDB)
        num_filters  : number of feature maps per filter size
        filter_sizes : tuple of n-gram window sizes to use in parallel
        dropout      : dropout probability before the final classifier
        pad_idx      : padding token index in the vocabulary
    """
    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int   = 128,
        num_classes:  int   = 2,
        num_filters:  int   = 100,
        filter_sizes: tuple = (3, 4, 5),
        dropout:      float = 0.5,
        pad_idx:      int   = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # One Conv2d per filter size; width == embed_dim so output is (batch, num_filters, L, 1)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(k, embed_dim))
            for k in filter_sizes
        ])

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x).unsqueeze(1)   # (batch, 1, seq_len, embed_dim)

        # Apply each filter, relu, then max-over-time pool
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(embedded)).squeeze(3)   # (batch, num_filters, L)
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  # (batch, num_filters)
            pooled.append(h)

        cat = torch.cat(pooled, dim=1)              # (batch, num_filters * len(filter_sizes))
        cat = self.dropout(cat)
        return self.classifier(cat)


# =============================================================================
# BERT - Smaller Variant (DistilBERT)
# =============================================================================
# DistilBERT is a distilled version of BERT-base with 6 layers instead of 12,
# 66M parameters vs 110M, and ~97% of BERT's GLUE performance.
# The HuggingFace interface is nearly identical to BertForSequenceClassification
# so no changes are needed in your training loop or compression utilities.
#
# Key interface difference from BERT:
#   DistilBERT does not use token_type_ids, so do not pass that argument
#   from your tokenizer output. Pass only input_ids and attention_mask.
#
#   tokenizer output usage:
#     encoding = tokenizer(texts, padding=True, truncation=True,
#                          max_length=512, return_tensors="pt")
#     logits = model(
#         input_ids=encoding["input_ids"],
#         attention_mask=encoding["attention_mask"]
#         # do NOT pass token_type_ids here
#     ).logits
# =============================================================================
 
def build_bert(num_classes: int = 2, pretrained: bool = True):
    """
    Returns a DistilBERT model configured for `num_classes` output classes.
 
    Replaces the original bert-base-uncased with distilbert-base-uncased:
      - 6 transformer layers (vs 12 in BERT-base)
      - 66M parameters (vs 110M in BERT-base)
      - Same 768-dimensional hidden states
      - No token_type_ids (see interface note above)
 
    Args:
        num_classes : number of output classes (2 for binary IMDB sentiment)
        pretrained  : if True, loads distilbert-base-uncased pretrained weights
 
    Returns:
        DistilBertForSequenceClassification with PyTorch backend
    """
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertConfig,
    )
 
    if pretrained:
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_classes,
        )
    else:
        config = DistilBertConfig(
            num_labels=num_classes,
            n_layers=6,          # DistilBERT default
            n_heads=12,          # DistilBERT default
            dim=768,             # same hidden size as BERT-base
            hidden_dim=3072,     # same FFN dim as BERT-base
        )
        model = DistilBertForSequenceClassification(config)
 
    return model


# =============================================================================
# BiLSTM - Bidirectional LSTM Sentiment Classifier
# =============================================================================
# The forward and backward LSTM passes each produce a hidden state vector at
# the end of the sequence.  Concatenating these two vectors gives the model
# context from both directions of the sentence before classifying.
#
# The vocabulary and pad_idx should be set from the same tokenizer used for
# TextCNN so that both models share the same token ID space.
# =============================================================================

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence classification.

    Args:
        vocab_size  : vocabulary size (set from dataset tokenizer)
        embed_dim   : word embedding dimensionality
        hidden_size : number of hidden units per direction
        num_layers  : number of stacked BiLSTM layers
        num_classes : number of output classes (2 for IMDB)
        dropout     : dropout probability applied to embeddings and between layers
        pad_idx     : padding token index (masks gradient through padding)
    """
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 128,
        hidden_size: int   = 256,
        num_layers:  int   = 2,
        num_classes: int   = 2,
        dropout:     float = 0.3,
        pad_idx:     int   = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        # hidden_size * 2 because forward + backward final states are concatenated
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))      # (batch, seq_len, embed_dim)
        _, (hn, _) = self.lstm(embedded)
        # hn: (num_layers * 2, batch, hidden_size)
        # Last layer forward state: hn[-2], backward state: hn[-1]
        final = torch.cat([hn[-2], hn[-1]], dim=1)      # (batch, hidden_size * 2)
        return self.classifier(self.dropout(final))
