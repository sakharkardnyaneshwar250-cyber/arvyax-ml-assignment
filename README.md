# ArvyaX ML Assignment

## Approach
- TF-IDF used for text representation
- Metadata features included (sleep, stress, energy)
- RandomForest used for prediction

## Tasks
- Emotional state → classification
- Intensity → regression

## Decision Engine
Hybrid rule-based system using:
- predicted state
- intensity
- stress
- energy
- time

## Uncertainty
- confidence = max probability
- uncertain if < 0.6

## Ablation Study
- Text only < Text + metadata

## Run
1. python train.py
2. python predict.py