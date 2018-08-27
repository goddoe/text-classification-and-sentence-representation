# Text Classification and Sentence Representation
PyTorch implementation of several sentence representation methods: CBOW, Relation Network(RN), CNN, Self Attention.

## Contents

+ CBOW
+ Relation Network (RN)
+ CNN
+ Self Attention


## Experiments
Training without fine-tuning (eg. parameter optimizer and pretraining).

### Common Settings.

+ Dataset: IMDB
+ learning rate: 0.001

Model|Epochs|Train Accuracy|Test Accuracy
---|---|---|---
CBOW|5|0.95|0.89
CNN|5|1.0|0.82
