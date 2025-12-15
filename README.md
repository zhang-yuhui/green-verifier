# GreenVerifier

GreenVerifier is a retrieval-augmented, encoder-based framework for automated verification of ESG claims in long-form corporate sustainability and TCFD reports.  
Given a claim and its source report, the system classifies the claim as **Supported** or **Not Supported**, while identifying the underlying textual or numerical evidence.

The pipeline consists of preprocessing and indexing of reports, dense retrieval of candidate evidence, and supervised verification using a DeBERTa cross-encoder trained with Multiple Instance Learning (MIL).

---

## Installation

```bash
conda create -n greenverifier python=3.10
conda activate greenverifier
pip install -r requirements.txt

