import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from ..ExtendSentenceTransformer import ExtendSentenceTransformer
from .. import util
import copy
import random
import math
from .. import InputExample
import numpy as np


class ContrastiveTensionLoss(nn.Module):
    """
        This loss expects as input a batch consisting of multiple mini-batches of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_{K+1}, p_{K+1})
        where p_1 = a_1 = a_2 = ... a_{K+1} and p_2, p_3, ..., p_{K+1} are expected to be different from p_1 (this is done via random sampling).
        The corresponding labels y_1, y_2, ..., y_{K+1} for each mini-batch are assigned as: y_i = 1 if i == 1 and y_i = 0 otherwise.
        In other words, K represent the number of negative pairs and the positive pair is actually made of two identical sentences. The data generation
        process has already been implemented in readers/ContrastiveTensionReader.py
        For tractable optimization, two independent encoders ('model1' and 'model2') are created for encoding a_i and p_i, respectively. For inference,
        only model2 are used, which gives better performance. The training objective is binary cross entropy.
        For more information, see: https://openreview.net/pdf?id=Ov_sMNau-PF

    """

    def __init__(self, model: ExtendSentenceTransformer):
        """
        :param model: SentenceTransformer model
        """
        super(ContrastiveTensionLoss, self).__init__()
        self.model1 = model 
        self.model2 = copy.deepcopy(model)
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        sentence_features1, sentence_features2 = tuple(sentence_features)
        reps_1 = self.model1(sentence_features1)['sentence_embedding']  
        reps_2 = self.model2(sentence_features2)['sentence_embedding']
        sim_scores = torch.matmul(reps_1[:, None], reps_2[:, :, None]).squeeze(-1).squeeze(
            -1)
        loss = self.criterion(sim_scores, labels.type_as(sim_scores))
        return loss

class ContrastiveTensionLossWithMasking(nn.Module):
    def __init__(self, model: nn.Module, contrastive_weight=1.0, masked_weight=1.0):
        super(ContrastiveTensionLossWithMasking, self).__init__()
        self.model1 = model
        self.model2 = copy.deepcopy(model)
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        
        self.masked_loss_fn = nn.CrossEntropyLoss(reduction='sum')

        self.contrastive_weight = contrastive_weight
        self.masked_weight = masked_weight
        
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        sentence_features1, sentence_features2 = tuple(sentence_features)

        output1 = self.model1(sentence_features1)
        output2 = self.model2(sentence_features2)
        reps_1 = output1['sentence_embedding']  
        reps_2 = output2['sentence_embedding'] 
        sim_scores = torch.matmul(reps_1[:, None], reps_2[:, :, None]).squeeze(-1).squeeze(-1)
        
        contrastive_loss = self.criterion(sim_scores, labels.type_as(sim_scores))

        mask_idx = sentence_features1.get('mask_idx', None)
        masked_labels = sentence_features1.get('masked_labels', None)

        if True:
            masked_logits = output1.get('masked_logits', None) 
            if masked_logits is not None:
                masked_loss = self.masked_loss_fn(masked_logits.view(-1, masked_logits.size(-1)), masked_labels.view(-1))
            else:
                masked_loss = 0.0
        else:
            masked_loss = 0.0

        total_loss = self.contrastive_weight*contrastive_loss + self.masked_weight * masked_loss

        return total_loss

class ContrastiveTensionLossInBatchNegatives(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim):
        """
        :param model: SentenceTransformer model
        """
        super(ContrastiveTensionLossInBatchNegatives, self).__init__()
        self.model1 = model 
        self.model2 = copy.deepcopy(model)
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(scale))

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        sentence_features1, sentence_features2 = tuple(sentence_features)
        embeddings_a = self.model1(sentence_features1)['sentence_embedding'] 
        embeddings_b = self.model2(sentence_features2)['sentence_embedding']
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.logit_scale.exp()
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(scores.t(), labels)) / 2

class ContrastiveTensionDataLoader:
    def __init__(self, sentences, batch_size, pos_neg_ratio=8):
        self.sentences = sentences
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        self.collate_fn = None

        if self.batch_size % self.pos_neg_ratio != 0:
            raise ValueError(
                f"ContrastiveTensionDataLoader was loaded with a pos_neg_ratio of {pos_neg_ratio} and a batch size of {batch_size}. The batch size must be devisable by the pos_neg_ratio")

    def __iter__(self):
        random.shuffle(self.sentences)
        sentence_idx = 0
        batch = []

        while sentence_idx + 1 < len(self.sentences):
            s1 = self.sentences[sentence_idx]
            if len(batch) % self.pos_neg_ratio > 0: 
                sentence_idx += 1
                s2 = self.sentences[sentence_idx]
                label = 0
            else:  
                s2 = self.sentences[sentence_idx]
                label = 1

            sentence_idx += 1
            batch.append(InputExample(texts=[s1, s2], label=label))

            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch) if self.collate_fn is not None else batch
                batch = []

    def __len__(self):
        return math.floor(len(self.sentences) / (2 * self.batch_size))

class ContrastiveTensionExampleDataLoader:
    def __init__(self, examples, batch_size, pos_neg_ratio=8, anisomorphic_in_batch=False):
        self.examples = examples
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        self.collate_fn = None
        self.aniso_in_batch = anisomorphic_in_batch
        if self.batch_size % self.pos_neg_ratio != 0:
            raise ValueError(
                f"ContrastiveTensionDataLoader was loaded with a pos_neg_ratio of {pos_neg_ratio} and a batch size of {batch_size}. The batch size must be devisable by the pos_neg_ratio")

    def __iter__(self):
        random.shuffle(self.examples)
        example_idx = 0
        batch = []

        while example_idx + 1 < len(self.examples):
            if len(batch) % self.pos_neg_ratio > 0:  
                example_idx += 1
                s2 = self.examples[example_idx]
                label = 0
                if s1.edge_index:
                    batch.append(InputExample(texts=[s1.texts[0], s2.texts[0]], label=label,
                                              edge_index=[s1.edge_index[0], s2.edge_index[0]],
                                              edge_type=[s1.edge_type[0], s2.edge_type[0]],
                                              pos_ids=[s1.pos_ids[0], s2.pos_ids[0]]))
                else:
                    batch.append(InputExample(texts=[s1.texts[0], s2.texts[0]], label=label))
                if len(batch) >= self.batch_size:
                    yield self.collate_fn(batch) if self.collate_fn is not None else batch
                    batch = []
                    example_idx += 1

            else: 
                s1 = self.examples[example_idx]

                label = 1
                s1.set_label(label)
                batch.append(s1)
                if len(batch) >= self.batch_size:
                    yield self.collate_fn(batch) if self.collate_fn is not None else batch
                    batch = []
                    example_idx += 1

    def __len__(self):
        return math.floor(len(self.examples) / (2 * self.batch_size))
class ContrastiveTensionExampleDataLoader_our:
    def __init__(self, examples, tokenizer, batch_size, max_seq_length, pos_neg_ratio=8, anisomorphic_in_batch=False):
        self.examples = examples
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.pos_neg_ratio = pos_neg_ratio
        self.aniso_in_batch = anisomorphic_in_batch

        if self.batch_size % self.pos_neg_ratio != 0:
            raise ValueError(f"The batch size must be divisible by pos_neg_ratio ({self.pos_neg_ratio})")

    

    def __iter__(self):
        random.shuffle(self.examples) 
        example_idx = 0 
        batch = [] 

        while example_idx + 1 < len(self.examples):
            if len(batch) % self.pos_neg_ratio > 0: 
                s2 = self.examples[example_idx] 
                label = 0 
                if hasattr(s1, 'edge_index'):
                    batch.append(InputExample(
                        texts=[s1.texts[0], s2.texts[0]], label=label,
                        edge_index=[s1.edge_index[0], s2.edge_index[0]],
                        edge_type=[s1.edge_type[0], s2.edge_type[0]],
                        pos_ids=[s1.pos_ids[0], s2.pos_ids[0]],
                        mask_idx=[s1.mask_idx[0],s2.mask_idx[0]],
                        masked_labels=[s1.masked_lables[0],s2.masked_lables[0]]
                    ))
                else:
                    batch.append(InputExample(texts=[s1.texts[0], s2.texts[0]], label=label))
                
                example_idx += 1
                if len(batch) >= self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []

            else: 
                s1 = self.examples[example_idx]
                label = 1 
                batch.append(InputExample(
                    texts=[s1.texts[0], s1.texts[0]],
                    label=label,
                    edge_index=[s1.edge_index[0], s1.edge_index[0]],
                    edge_type=[s1.edge_type[0], s1.edge_type[0]],
                    pos_ids=[s1.pos_ids[0], s1.pos_ids[0]],
                    mask_idx=[s1.mask_idx[0], s1.mask_idx[0]], 
                    masked_labels=[s1.masked_labels[0], s1.masked_labels[0]]
                ))
                example_idx += 1

                if len(batch) >= self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []

    def __len__(self):
        return len(self.examples) // self.batch_size 