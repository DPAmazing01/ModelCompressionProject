"""
models/__init__.py
------------------
Clean import interface for all 9 models across 3 tasks.

Usage
-----
    from models import M3, build_ast, AudioLSTM          # audio classification
    from models import TextCNN, build_bert, BiLSTM        # sentiment analysis
    from models import GRUSeq2Seq, TranslationTransformer, ConvS2S  # translation

Quick-build examples
--------------------
    # Audio Classification (Speech Commands, 35 classes)
    m3_model        = M3(num_classes=35)
    ast_model       = build_ast(num_classes=35, pretrained=True)
    audio_lstm      = AudioLSTM(input_size=64, hidden_size=128, num_classes=35)

    # Sentiment Analysis (IMDB, 2 classes)
    text_cnn        = TextCNN(vocab_size=25000, embed_dim=128, num_classes=2)
    bert_model      = build_bert(num_classes=2, pretrained=True)
    bilstm_model    = BiLSTM(vocab_size=25000, embed_dim=128, num_classes=2)

    # Machine Translation (IWSLT2017 de-en, vocab sizes from dataset)
    gru_model       = GRUSeq2Seq(src_vocab_size=8000, tgt_vocab_size=6000)
    transformer     = TranslationTransformer(src_vocab_size=8000, tgt_vocab_size=6000)
    convs2s         = ConvS2S(src_vocab_size=8000, tgt_vocab_size=6000)
"""

from .audio_models import (
    M3,
    build_ast,
    AudioLSTM,
)

from .sentiment_models import (
    TextCNN,
    build_bert,
    BiLSTM,
)

from .translation_models import (
    GRUSeq2Seq,
    TranslationTransformer,
    ConvS2S,
)

__all__ = [
    # Audio
    "M3",
    "build_ast",
    "AudioLSTM",
    # Sentiment
    "TextCNN",
    "build_bert",
    "BiLSTM",
    # Translation
    "GRUSeq2Seq",
    "TranslationTransformer",
    "ConvS2S",
]
