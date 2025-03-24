import json
import logging
import torch
from tqdm import tqdm
import collections
import os
import random
import matplotlib.pyplot as plt
from sentence_transformers import ExtendSentenceTransformer, LoggingHandler, models, InputExample
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import torch.nn as nn
import torch.nn.functional as F
import datetime
from preprocess import generate_ref_edge, generate_wiki_edge

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
def log_message(message):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

model_name = 'bert-base-uncased'
model_type = model_name.split('/')[-1]
batch_size = 16
pos_neg_ratio = 4  
epochs = 1
max_seq_length = 128
gnn = 'GraphConv'
gnn_layer = 4
adapter_size = 768
learning_rate = 1e-5
add_graph = True
model_save_path = 'output/ct-debug-bert'
os.makedirs(model_save_path, exist_ok=True)
log_message(f"Saving model to {model_save_path}")

log_message("Starting data preprocessing...")

wikipedia_dataset_path = 'Semantic-consistency/data/train.json'
train_samples = []
with open(wikipedia_dataset_path, 'r', encoding='utf8') as fIn:
    lines = fIn.readlines()
    for line in tqdm(lines):
        line = json.loads(line)
        if add_graph:
            graph_triples = line["aligned_triples"]
            if graph_triples == []: continue
            edge_index, edge_type, pos_ids, mask_idx,masked_labels = generate_wiki_edge(graph_triples, max_seq_length)
            if edge_index[0] is None:
                continue
            inp_example = InputExample(texts=[line['amr_simple'], line['amr_simple']],
                                       edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids,
                                       mask_idx=mask_idx,
                                       masked_labels=masked_labels)
            train_samples.append(inp_example)
        else:
            inp_example = InputExample(texts=[line['amr_simple'], line['amr_simple']])
            train_samples.append(inp_example)

log_message("Data preprocessing completed.")
log_message("Initializing model...")


word_embedding_model = models.ExtendTransformer(model_name, max_seq_length=max_seq_length, adapter_size=adapter_size,
                                               gnn=gnn, gnn_layer=gnn_layer, add_gnn=add_graph)
tokenizer = word_embedding_model.tokenizer
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = ExtendSentenceTransformer(modules=[word_embedding_model, pooling_model])

train_dataloader = losses.ContrastiveTensionExampleDataLoader(train_samples, batch_size=batch_size,
                                                              pos_neg_ratio=pos_neg_ratio)

log_message("Preparing validation data...")
dev_sts_dataset_path = 'Semantic-consistency/data/dev-sense1.json'
dev_samples = []

with open(dev_sts_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        score = float(line['score']) / 5.0 
        if add_graph:
            edge_index, edge_type, pos_ids = generate_ref_edge(line, tokenizer, max_seq_length)
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score, edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
        else:
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score)
        dev_samples.append(inp_example)

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

train_loss = losses.ContrastiveTensionLossWithMasking(model)


log_message("Starting training...")


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=epochs,
    evaluation_steps=1000,
    weight_decay=0,
    warmup_steps=0,
    optimizer_class=torch.optim.RMSprop,
    optimizer_params={'lr': learning_rate},
    output_path=model_save_path,
    use_amp=False,
)
log_message("Training completed.")
print(f"Saving model to {model_save_path}")
model.save(model_save_path)  #



