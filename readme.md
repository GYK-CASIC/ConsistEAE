# ConsistEE

This repository contains the code for our paper: ConsistEE: Retriving Demonstraions with Consist Semantic and Lexical for Low-Resource Event Extraction.

## Environment
### semantic
- numpy==1.21.6
- torch==1.12.0
- torch_geometric==2.1.0
- transformers==4.24.0
- networkx==2.6.3
- unidecode==1.3.6
- nltk==3.7
### syntactic
- torch==2.5.1
- stanza==1.9.2
- numpy==2.0.2
- networkx==3.2.1
- requests==2.32.3
- tqdm==4.67.1
### EAE
- torch==2.0.1
- torchaudio==2.0.2
- transformers==4.41.2
- sentence-transformers==3.2.0
- spacy==3.7.2
- nltk==3.8.1
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.1.2
- requests==2.32.3
- flair==0.13.1
- tqdm==4.66.1

## Datasets
We support `ace05e`, `ace05ep`

Our preprocessing mainly adapts [DEGREE](https://github.com/PlusLabNLP/DEGREE) released scripts with minor modifications. We deeply thank the contribution from the authors of the paper.

## Three stage of ConsistEAE

### Pre-Extraction with LLM

First, K demonstrations of the same event type are manually selected for each event type and input into LLM to pre-extract the training dataset. Both the demonstrations and the pre-extracted dataset are placed in the `EAE/pre-extraction` folder.

```
conda activate EAE
python EAE/experiment.py
```


###  Demonstration Selection with Linguistic Consistency

* **Input** 
  1. Pre-extracted dataset from LLM
  2. Test dataset
* **Output**
	1. Demonstrations based on semantic consistency
	2. Demonstrations based on syntactic consistency
	3. Demonstrations based on linguistic consistency

These prompt data will be input into LLM for event argument extraction.

#### Semantic Consistency Measurement
#### We support training a model that captures both global and local semantic consistency. The trained model will be stored in the `output/ct-debug-bert` directory.

```
conda activate semantic
python Semantic-consistency/sentence-transformers/train_stsb_ct_amr.py
```

After training is completed, use the following testing script for evaluation:
```
python Semantic-consistency/sentence-transformers-test/test_amrsim.py
```
This will output demonstrations using semantic consistency, which will be saved in `Semantic-consistency/data/ACE05EP/prompt/prompt_EEA_with_trigger.json`. Additionally, consistency scores between sentence pairs will be saved in the `Syntactic-consistency/ace05ep/similarity_scores.txt` file.


#### Syntactic Consistency Measurement

Run the following code to obtain demonstrations based on syntactic consistency:

```
conda activate syntactic
python Syntactic-consistency/prompt_gen_hanlp_ace05ep.py
```
The output will be saved as `Syntactic-consistency/ace05ep/prompt_EEA_with_trigger.json`. This file contains the generated prompts, which can be input into the large pre-trained model for event argument extraction tasks.

#### Linguistic Consistency Measurement

The weighted semantic consistency score and syntactic consistency score are used to obtain demonstrations based on linguistic consistency. The output will be saved as `Syntactic-consistency/ace05ep/prompt_EEA_with_trigger_merge.json`.

```
conda activate syntactic
python Syntactic-consistency/merge.py
```

## Event Argument Extraction with ICL

Extraction by LLM
```
conda activate EAE
python EAE/experiment.py
```

Head span word extraction:
```
python EAE/process.py
python EAE/head.py
```
Evaluation:

```
python EAE/evaluate.py
```
## Acknowledgments and Citations

This project borrows or uses code from the following projects and resources, for which we are grateful:

- [AMRSim](https://github.com/zzshou/AMRSim)
- [Sentence-Transformer](https://www.sbert.net/)
- [stanza](https://stanfordnlp.github.io/stanza/download_models.html)

We are very grateful for the contributions of the authors of the above projects.

If you find that the code is useful in your research, please consider citing our paper.

  title={ConsistEE: Retriving Demonstraions with Consist Semantic and Lexical for Low-Resource Event Extraction},
  author={Yikai Guo, Xuemeng Tian, Bin Ge, Yuting Yang, Yao He, Wenjun Ke, Junjie Hu, Yanyang Li, Haoran Luo},
  year={2025}
}


