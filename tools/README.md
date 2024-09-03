# Tools for morphological analysis

As suggested in [[1](#references)], the textual answers returned while evaluating models using the SQuAD and SQAD datasets are processed using morphological analysis, in order to compensate for the large morphological richness of the Czech language compared to English.

Each word of the returned answer, as well as the reference answers, is first lemmatized using MorphoDita. These lemmas are then replaced by the roots of their word-formation relation trees according to the Derinet lexicon.

## MorphoDita

MorphoDita and its underlying models can be obtained [here](https://ufal.mff.cuni.cz/morphodita). The Czech MorfFlex2 model used during the evaluation is included in this repo.

The MorphoDita models are licensed under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## DeriNet

DeriNet can be obtained from [GitHub](https://github.com/vidraj/derinet). Only the 2.0 release of the lexicon is provided in this repository.

Derinet is licensed under [Creative Commons BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)

## References

[1] Macková and Straka, [Reading Comprehension in Czech via Machine Translation and Cross-lingual Transfer](https://browse.arxiv.org/pdf/2007.01667.pdf), 2020

[2] Straková et al., [Open-Source Tools for Morphology, Lemmatization, POS Tagging and Named Entity Recognition](https://aclanthology.org/P14-5003), ACL 2014

[3] Vidra et al., [DeriNet 2.0: Towards an All-in-One Word-Formation Resource](https://aclanthology.org/W19-8510), 2019
