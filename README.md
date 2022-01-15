# Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning

- This repository contains the code to run the experiments of our paper:

Alberto Maria Metelli, Alessio Russo, and Marcello Restelli<br>
**Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning**<br>
In Advances in Neural Information Processing Systems 34 (NeurIPS). 2021. [[Paper]](https://papers.nips.cc/paper/2021/hash/4476b929e30dd0c4e8bdbcc82c6ba23a-Abstract.html)

- The code builds upon the **Open Bandit Pipeline** [https://github.com/st-tech/zr-obp](https://github.com/st-tech/zr-obp)

## Running Experiments

#### Off-Policy Evaluation Synthetic Example
```
python3.7 examples/off_evaluation_gaussian.py 
```
#### Off-Policy Evaluation UCI Datasets
- To run these experiments it is necessary to have a directory "datasets" in the root of the project with the UCI datasets downloaded (see also obp/dataset/uci_datasets.py)
- To run the comprehensive experiment over the 110 configurations
```
python3.7 examples/examples_with_uci/off_evaluation_all.py
```
- To run an experiment on a single dataset
```
python3 examples/examples_with_uci/off_evaluation_one.py
```

#### Off-Policy Learning UCI Datasets
```
python3.7 examples/examples_with_uci/off_learning_one.py

```

# Citation
If you use this code in your work, please cite our paper:

Bibtex:
```
@incollection{metelli2021subgaussian,
	author    = {Alberto Maria Metelli and Alessio Russo and Marcello Restelli},
	title	  = {Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning},
	booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
	year      = {2021},
}
```

# License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
