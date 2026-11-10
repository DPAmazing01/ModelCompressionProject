"""
audio_models.py
---------------
Three models for audio classification on the Google Speech Commands dataset
(35-class spoken word recognition).

  M3         - 3-layer 1D CNN operating on raw waveforms
  AST        - Audio Spectrogram Transformer via HuggingFace
  AudioLSTM  - LSTM operating on mel-spectrogram frame sequences

Input conventions
  M3        : (batch, 1, n_samples)         raw waveform, 16 kHz -> 16000 samples
  AST       : (batch, n_mels, time_frames)  mel-spectrogram from ASTFeatureExtractor
  AudioLSTM : (batch, time_steps, n_mels)  mel-spectrogram as time-first sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# M3 - 3-Layer 1D CNN on Raw Waveform
# =============================================================================
# Derived from the M5 architecture (Dai et al., 2017) by reducing the number
# of convolutional blocks from 5 to 3.  A wide first kernel captures low-level
# acoustic features from the raw signal; subsequent layers learn higher-level
# patterns.  AdaptiveAvgPool1d at the end makes the model resolution-agnostic.
# =============================================================================

class M3(nn.Module):
    """
    Lightweight 3-layer 1D CNN for raw-waveform audio classification.

    Args:
        num_classes : number of output classes (35 for Speech Commands)
        n_channel   : number of filters in the final conv layer;
                      earlier layers scale proportionally
    """
    def __init__(self, num_classes: int = 35, n_channel: int = 256):
        super().__init__()

        # Block 1: wide kernel to capture raw acoustic structure
        self.conv1 = nn.Conv1d(1, 32, kernel_size=80, stride=16)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        # Block 2: intermediate feature extraction
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)

        # Block 3: high-level representation
        self.conv3 = nn.Conv1d(64, n_channel, kernel_size=3)
        self.bn3   = nn.BatchNorm1d(n_channel)
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # collapses temporal dimension

        self.classifier = nn.Linear(n_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_samples)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)               # (batch, n_channel)
        return self.classifier(x)


# =============================================================================
# AST - Audio Spectrogram Transformer (Small Variant)
# =============================================================================
# Wraps HuggingFace ASTForAudioClassification.
# Treating the mel-spectrogram as a grid of patches (analogous to ViT for
# images), AST applies standard Transformer self-attention over those patches.
#
# Preprocessing:
#   Use ASTFeatureExtractor from transformers to convert raw waveforms to
#   mel-spectrograms before passing to this model:
#
#     from transformers import ASTFeatureExtractor
#     extractor = ASTFeatureExtractor.from_pretrained(
#         "MIT/ast-finetuned-speech-commands-v2"
#     )
#     inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt")
#     logits = ast_model(**inputs).logits
# =============================================================================
 
def build_ast(num_classes: int = 35, pretrained: bool = True):
    """
    Returns a smaller AST model configured for `num_classes` output classes.
 
    Uses MIT/ast-finetuned-audioset-10-10-0.4593 as the pretrained base,
    which is lighter than the Speech Commands v2 checkpoint and fine-tunes
    well on Speech Commands with a replaced classification head.
 
    Args:
        num_classes : number of output classes (35 for Speech Commands)
        pretrained  : if True, loads the lighter pretrained AST weights
 
    Returns:
        ASTForAudioClassification with replaced classification head
    """
    from transformers import ASTForAudioClassification, ASTConfig
 
    if pretrained:
        model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        # Always replace the head since AudioSet has 527 classes
        # and Speech Commands has 35
        in_features = model.classifier.dense.in_features
        model.classifier.dense = nn.Linear(in_features, num_classes)
    else:
        # Manually configure a smaller AST from scratch
        config = ASTConfig(
            num_labels=num_classes,
            hidden_size=384,          # DeiT-Small equivalent hidden dim
            num_hidden_layers=6,      # half the layers of the default AST
            num_attention_heads=6,    # 384 / 6 = 64 per head
            intermediate_size=1536,   # 4 * hidden_size convention
        )
        model = ASTForAudioClassification(config)
 
    return model


# =============================================================================
# AudioLSTM - LSTM on Mel-Spectrogram Frames
# =============================================================================
# The mel-spectrogram is treated as a temporal sequence of feature vectors
# (one vector per time frame), which maps naturally to LSTM input.
# The final hidden state of the last LSTM layer is used for classification.
#
# Preprocessing:
#   Compute mel-spectrogram with torchaudio.transforms.MelSpectrogram, then
#   convert to log scale and transpose to (time_steps, n_mels):
#
#     mel = torchaudio.transforms.MelSpectrogram(
#         sample_rate=16000, n_mels=64
#     )(waveform)                                # (1, n_mels, time)
#     log_mel = (mel + 1e-9).log2().squeeze(0)   # (n_mels, time)
#     sequence = log_mel.T                       # (time, n_mels)  -> batched: (batch, time, n_mels)
# =============================================================================

class AudioLSTM(nn.Module):
    """
    Multi-layer LSTM classifier for mel-spectrogram sequences.

    Args:
        input_size  : number of mel filter banks (default 64)
        hidden_size : number of LSTM hidden units per direction
        num_layers  : number of stacked LSTM layers
        num_classes : number of output classes
        dropout     : dropout probability applied between LSTM layers
    """
    def __init__(
        self,
        input_size:  int = 64,
        hidden_size: int = 128,
        num_layers:  int = 2,
        num_classes: int = 35,
        dropout:     float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time_steps, n_mels)
        _, (hn, _) = self.lstm(x)
        # hn: (num_layers, batch, hidden_size) — take the last layer
        out = self.dropout(hn[-1])   # (batch, hidden_size)
        return self.classifier(out)
