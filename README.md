# Text Classification and Sentence Representation
PyTorch implementation of several sentence representation methods: CBOW, Relation Network(RN), CNN, Self Attention.

## Contents

+ CBOW
+ Relation Network (RN)
+ CNN
+ Self Attention


## Experiments
Experiments without fine-tuning.

### Common Settings.

+ Dataset: IMDB
+ learning rate: 0.001

Model|Epochs|Train Accuracy|Test Accuracy
---|---|---|---
CBOW|5|0.95|0.89
RN|5|1.0|0.78
CNN|5|1.0|0.82
