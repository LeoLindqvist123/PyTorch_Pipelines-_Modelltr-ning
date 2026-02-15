# Labb 2: CIFAR-10 Klassificering

## Dataset
CIFAR-10: 50,000 träningsbilder, 10,000 testbilder (flygplan, bilar, fåglar, etc.)

## Modell
Convolutional Neural Network (CNN):
- Conv Layer 1: 32 filters (3×3)
- Conv Layer 2: 64 filters (3×3)
- Conv Layer 3: 128 filters (3×3)
- Fully Connected: 256 → 10 klasser

## Experiment

| Experiment | Epochs | Batch Size | Learning Rate | Test Accuracy |
|-----------|--------|------------|---------------|---------------|
| 1 (Baseline) | 20 | 32 | 0.001 | 72.26% |
| 2 (Högre LR) | 20 | 32 | 0.01 | 10.06% |
| 3 (Större batch) | 20 | 64 | 0.001 | **73.31%** |

## Slutsats
Experiment 3 gav bäst resultat (73.31%). För hög learning rate (exp 2) förstörde träningen. CNN-arkitekturen förbättrade accuracy från ~50% (FC-nätverk) till 73%, vilket visar vikten av att använda rätt modelltyp för bilddata.

## Kör projektet
```bash
uv run python main.py
```