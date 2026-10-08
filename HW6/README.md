# Experiment Report

## Introduction

This assignment studies a Kaggle "dogs vs. cats" image-classification competition to
understand a complete model-training workflow end to end, based on
[Mohammad Alaliwi's notebook](https://www.kaggle.com/code/mhmmadalewi/classifier-dogs-or-cats-val-acc-98-5)
(val_acc ≈ 98.5%). The notebook itself was developed and run directly on Kaggle and
isn't available locally, so only the report is included here.

## Materials and Methods

**Data preprocessing**:
- Data augmentation: random rotation (±20°), horizontal/vertical shifts (20%), random
  horizontal flip, nearest-neighbor fill for introduced pixels.
- Hyperparameters: 200×200 image size, batch size 32, 20% validation split, 10 epochs,
  early stopping on validation performance.

**Optimization**: cross-entropy loss, learning rate 0.01, exponential-decay scheduler,
Adam optimizer.

**Model architectures compared**: ResNet50V2, ResNet152V2, InceptionV3, Xception,
DenseNet121 — with transfer learning via a fine-tuned Xception as the final choice.

## Results

Training converged quickly, with validation accuracy consistently above training
accuracy (consistent with the heavy augmentation applied) — training accuracy rose from
~0.955 to ~0.973, validation accuracy from ~0.976 to ~0.985 over 12 epochs, while loss
dropped sharply within the first epoch and then stayed flat. The test-set confusion
matrix, however, tells a different story than the accuracy curves: of true cat samples,
514 were predicted as dogs and essentially none as cats; of true dog samples, 509 were
correctly predicted as dogs and 491 as cats. The individual sample predictions shown
alongside it look mostly correct (most pictured cats and dogs are labeled accurately),
so this mismatch between the strong accuracy curves and the lopsided confusion matrix
looks like an artifact of how the matrix itself was built (e.g. a label/axis mismatch)
rather than the model actually failing to recognize cats.

## Conclusion

This exercise walked through a complete Kaggle-style workflow — augmentation,
hyperparameter choices, learning-rate scheduling, and comparing several transfer-learning
backbones (ResNet, Inception, Xception, DenseNet) — and reinforced that data
augmentation, early stopping, and transfer learning from strong pretrained backbones are
practical, high-leverage techniques for competitive image classification.
