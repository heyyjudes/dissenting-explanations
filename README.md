# Dissenting Explanations

This repo accompanies the paper: 
[Dissenting Explanations: Leveraging Disagreement to Reduce Model Overreliance](https://arxiv.org/abs/2307.07636)

### Data and Preprocessing
We use data from the [Folktables](https://github.com/socialfoundations/folktables) and 
[Opinion Spam Dataset](https://myleott.com/op-spam.html). 
The preprocessing script: [data_processing.py](data_processing.py)  can be used to preprocess 
the opinion spam dataset.

### Generating Dissenting Explanations
Global dissenting explanations can be generated using the [counter_exp_global.ipynb](counter_exp_global.ipynb) notebook 
and local dissenting explanations can be generated using [test-time-local-nn.ipynb](test-time-local-nn.ipynb)
and [test-time-local-svm.ipynb](test-time-local-svm.ipynb). 

### Experiment Materials
For our human experiments, we include the questions we used: 
[final_questions.csv](final_questions.csv). 

### Citation
```
@inproceedings{reingold2024dissenting,
title={Dissenting Explanations: Leveraging Disagreement to Reduce Model Overreliance},
author={Reingold, Omer and Shen, Judy Hanwen and Talati, Aditi},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
volume={38},
number={19},
pages={21537--21544},
year={2024}
}
```
