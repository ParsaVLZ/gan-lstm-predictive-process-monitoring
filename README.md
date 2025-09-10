# GAN-LSTM for Next Event & Timestamp Prediction (Process Mining)

This repository reproduces and extends **Predictive Business Process Monitoring** 
by applying hybrid **LSTM + GAN** architectures to two core tasks:

1. **Next Event Prediction** – predicting the most likely next activity in a process trace.  
2. **Timestamp Prediction** – estimating the time of the next event.  

Our implementation builds upon the ideas of Taymouri et al. and evaluates models 
on widely used public process-mining datasets (e.g., **Helpdesk**).

---

## Why this repo?
- GAN-based sequence modeling with LSTM backbone.
- Support for both **event prediction** and **timestamp prediction** tasks.
- Early Stopping + validation-based checkpoint selection.
- Dataset-adaptive prefix length configuration.
- Reported metrics: primarily **Top-1 Test Accuracy** (for events) and **MAE** (for timestamps).

---

## Repository Structure