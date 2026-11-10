(Project was worked on from February-April 2026) Project is a mini-research project that focuses on exploring 2 model compression techniques (Pruning & Low-Rank Factorization) across 3 different tasks (Audio Classification, Sentiment Analysis, and Machine Translation) using existing CNN and RNN models that have been finetuned for these tasks. The models and dataset used for the 3 tasks were:

• Audio Classification - M3 and AudioLSTM with the Google Speech Commands dataset
• Sentiment Analysis - TextCNN and BiLSTM with the IMDB dataset
• Machine Translation - ConvS2S and GRUSeq2Seq with the IWSLT2017 dataset

Experiments were conducted on the NVIDIA GeForce RTX 2080 Ti GPU (Accessed through the NCSU VCL), and the text file contains the training parameters used for the 6 models tested in the experiments. Additionally, due to computing constraints on the GPU used, the original models were only trained for 5 epochs and the compressed models were only retrained for 1 epoch. (Experiments results are in the CSV file included with this project)
