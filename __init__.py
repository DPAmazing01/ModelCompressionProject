"""
models/__init__.py
------------------
Clean import interface for all 6 models across 3 tasks.

Usage
-----
    from models import M3, AudioLSTM          # audio classification
    from models import TextCNN, BiLSTM        # sentiment analysis
    from models import GRUSeq2Seq, ConvS2S  # translation

Quick-build examples
--------------------
    # Audio Classification (Speech Commands, 35 classes)
    m3_model        = M3(num_classes=35)
    audio_lstm      = AudioLSTM(input_size=64, hidden_size=128, num_classes=35)

    # Sentiment Analysis (IMDB, 2 classes)
    text_cnn        = TextCNN(vocab_size=25000, embed_dim=128, num_classes=2)
    bilstm_model    = BiLSTM(vocab_size=25000, embed_dim=128, num_classes=2)

    # Machine Translation (IWSLT2017 de-en, vocab sizes from dataset)
    gru_model       = GRUSeq2Seq(src_vocab_size=8000, tgt_vocab_size=6000)
    convs2s         = ConvS2S(src_vocab_size=8000, tgt_vocab_size=6000)
"""

from .audio_models import (
    M3,
    AudioLSTM,
)

from .sentiment_models import (
    TextCNN,
    BiLSTM,
)

from .translation_models import (
    GRUSeq2Seq,
    ConvS2S,
)

__all__ = [
    # Audio
    "M3",
    "AudioLSTM",
    # Sentiment
    "TextCNN",
    "BiLSTM",
    # Translation
    "GRUSeq2Seq",
    "ConvS2S",
]
