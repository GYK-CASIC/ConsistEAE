# AMRSim

This repository contains the code for our ACL-2023
paper: [Evaluate AMR Graph Similarity via Self-supervised Learning](https://aclanthology.org/2023.acl-long.892/).
AMRSim collects silver AMR graphs and utilizes self-supervised learning methods to evaluate the similarity of AMR
graphs. 
AMRSim calculates the cosine of contextualized token embeddings, making it alignment-free.

## Requirements

Run the following script to install the dependencies:

```
pip install -r requirements.txt
```
Install [amr-utils](https://github.com/ablodge/amr-utils):
```
git clone https://github.com//ablodge/amr-utils
pip install penman
pip install ./amr-utils
```

## Computing AMR Similarity

### Preprocess

Linearize AMR graphs and calculate the relative distance of nodes from the root:

```
# 为了AMR图的相似性，首先需要将AMR图线性化，并计算节点相对于根节点的相对于根节点的相对距离 使用amr2json进行数据预处理
cd preprocess
python preprocess/amr2json.py -src /data01/zhanghang/txm/AMR/AMRSim-main/data/src.amr -tgt /data01/zhanghang/txm/AMR/AMRSim-main/data/tgt.amr
```

### Returning Similarity
# 在计算相似度之前，需下载预训练模型，并将其解压到输出目录下。然后运行test_amrsim来进行AMR的相似度计算
Download the model from [Google drive](https://drive.google.com/file/d/1klTrvv3hpIPxaCoMbRI7IJDme-Vq3UPS/view?usp=share_link) and
unzip to the output directory (/sentence-transformers/output/).

```
cd sentence-transformers
python test_amrsim.py
```


```
为了训练AMRSim 作者使用了AMR-DA提供的语料库，将从英文维基百科中随机抽取的一百万句子解析成AMR图（使用 SPRING 工具）。生成的维基百科 AMR 图经过预处理
```
### Training
Following data preparation in AMR-DA (Shou et al., 2022), AMRSim utilized SPRING (Bevilacqua et al., 2021) to parse [one-million sentences](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/tree/main) randomly sampled from English Wikipedia2 to AMR graphs. 

Generated Wiki AMR graphs were preprocessed and can be download from the [Google drive](https://drive.google.com/file/d/1VAuqLi0LsaCCbII_s2dPa9eDARicw18G/view?usp=sharing).
For training, run:
```
python sentence-transformers/train_stsb_ct_amr.py
```

## Citation

```
@inproceedings{shou-lin-2023-evaluate,
    title = "Evaluate {AMR} Graph Similarity via Self-supervised Learning",
    author = "Shou, Ziyi  and
      Lin, Fangzhen",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.892",
    pages = "16112--16123",
}
```


## Acknowledgments
This project uses code from the following open source projects:
- [AMR-DA](https://github.com/zzshou/amr-data-augmentation)
- [FactGraph](https://github.com/amazon-science/fact-graph)
- [Sentence-Transformers](https://www.sbert.net)

Thank you to the contributors of these projects for their valuable contributions to the open source community.

