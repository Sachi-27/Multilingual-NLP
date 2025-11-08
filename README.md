# Paper: Cross-lingual Transfer Dynamics in BLOOMZ

This repository contains code and datasets for **multilingual logit lens probing** analysis in multilingual language models.  
It accompanies the paper:

> **Cross-lingual Transfer Dynamics in BLOOMZ: Insights into Multilingual Generalization**  
> *Sabyasachi Samantaray and Preethi Jyothi, MRL 2025, EMNLP (ACL Anthology: [2025.mrl-main.4](https://aclanthology.org/2025.mrl-main.4/))*

## 🧭 About
The core idea is to analyze how multilingual models (like BLOOMZ) adapt to cross lingual transfer after instruction tuning. We do this using multilingual logit lens probing  across decoder layers. Using logit lens, the project aimed to uncover whether multilingual alignment is uniform across languages or influenced by certain triggers.

Key insights from the paper
- Latent suppression of interfering languages in instruction-tuned models improves performance in the target language. For example, fine-tuning on just 103 Russian machine-translated samples more than doubles Odia XQuAD performance. 
- Self-instruction tuning often degrades performance due to overgeneration, except for extremely low-resource languages like Odia. For short-span QA tasks, self-instruction leads to unnecessary verbose outputs.
- Instruction tuning in high-resource languages is not always beneficial for non-English-centric multilingual models.
- Higher data quality in training datasets leads to better cross-lingual transfer and downstream performance.

---

## 📂 Repository Structure

| File / Folder | Description |
|----------------|--------------|
| `datasets/` | Contains multilingual datasets used for probing and QA experiments. |
| `flores_lang_map.json` | JSON mapping of FLORES-style language codes to readable names or internal IDs; used in `probe_logit_lens.py`. |
| `QAtraining.py` | Script for training (fine-tuning) multilingual question-answering samples. |
| `QAinference.py` | Runs inference using trained QA models on multilingual data. |
| `logit_lens_tokenid_to_lang.py` | Maps model token IDs to language probabilities for logit-lens style probing. |
| `probe_logit_lens.py` | Performs probing analysis using the logit-lens method, mapping logits to language codes using `flores_lang_map.json`. |

## Citation
If you use this repository or reproduce the experiments, please cite:
```
@inproceedings{samantaray-jyothi-2025-cross,
  title = "Cross-lingual Transfer Dynamics in {BLOOMZ}: Insights into Multilingual Generalization",
  author = "Samantaray, Sabyasachi  and  Jyothi, Preethi",
  booktitle = "Proceedings of the 5th Workshop on Multilingual Representation Learning (MRL 2025)",
  pages = "47--61",
  year = "2025",
  address = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.mrl-main.4/"
}

```
