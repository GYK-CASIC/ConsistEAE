import json
import logging
import torch
from tqdm import tqdm
import collections
from sentence_transformers import ExtendSentenceTransformer, LoggingHandler, models, InputExample
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from preprocess import generate_ref_edge, generate_wiki_edge

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Training parameters
model_name = 'Semantic-consistency/facebook/bert'
batch_size = 16
pos_neg_ratio = 4  # batch_size must be divisible by pos_neg_ratio
epochs = 1
max_seq_length = 128

gnn = 'GraphConv'
gnn_layer = 4
adapter_size = 128  # Adapter size for GNN
learning_rate = 1e-5
add_graph = True  # Whether to use graph structure
model_save_path = 'Semantic-consistency/sentence-transformers/output/ct-debug-bert'  # Save model here

# Load and process training data (Wiki AMR data)
wikipedia_dataset_path = 'Semantic-consistency/data/train.json'
train_samples = []
with open(wikipedia_dataset_path, 'r', encoding='utf8') as fIn:
    lines = fIn.readlines()
    for line in tqdm(lines):
        line = json.loads(line)
        if add_graph:
            graph_triples = line["aligned_triples"]
            if graph_triples == []:
                continue
            # Generate graph structure from AMR triples
            edge_index, edge_type, pos_ids = generate_wiki_edge(graph_triples, max_seq_length)
            if edge_index[0] is None:
                continue
            inp_example = InputExample(texts=[line['amr_simple'], line['amr_simple']],
                                       edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
            train_samples.append(inp_example)
        else:
            inp_example = InputExample(texts=[line['amr_simple'], line['amr_simple']])
            train_samples.append(inp_example)


# Prepare DataLoader for contrastive loss training
train_dataloader = losses.ContrastiveTensionExampleDataLoader(train_samples, batch_size=batch_size, pos_neg_ratio=pos_neg_ratio)

# Initialize the SBERT model with BERT + GNN
word_embedding_model = models.ExtendTransformer(model_name, max_seq_length=max_seq_length, adapter_size=adapter_size,
                                                gnn=gnn, gnn_layer=gnn_layer, add_gnn=add_graph,from_tf=True)
# Check number of parameters in the model
counter = collections.Counter()
for name, param in word_embedding_model.named_parameters():
    counter[name.split('.')[0]] += torch.numel(param)

# Pooling layer to aggregate token-level embeddings to sentence-level embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# Combine transformer model and pooling model into one
model = ExtendSentenceTransformer(modules=[word_embedding_model, pooling_model])

# Load STS Benchmark development set for evaluation
dev_sts_dataset_path = 'Semantic-consistency/data/stsbenchmark/dev-sense.json'
dev_samples = []
with open(dev_sts_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        score = float(line['score']) / 5.0  # Normalize to range [0, 1]
        if add_graph:
            edge_index, edge_type, pos_ids = generate_ref_edge(line, word_embedding_model.tokenizer, max_seq_length)
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score, edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
        else:
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score)
        dev_samples.append(inp_example)

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# ContrastiveTensionLoss for training
train_loss = losses.ContrastiveTensionLoss(model)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=epochs,
    evaluation_steps=1000,  # Evaluate every 1000 steps
    weight_decay=0,
    warmup_steps=0,
    optimizer_class=torch.optim.RMSprop,
    optimizer_params={'lr': learning_rate},
    output_path=model_save_path,  # Save model here
    use_amp=False  # Set to True if using FP16
)

print(f"Model training completed. Saved at {model_save_path}")

# Evaluate the model on the test set after training
test_sts_dataset_path = 'Semantic-consistency/data/src_tgt.json'
test_samples = []
with open(test_sts_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        score = float(line['score']) / 5.0  # Normalize to range [0, 1]
        if add_graph:
            edge_index, edge_type, pos_ids = generate_ref_edge(line, word_embedding_model.tokenizer, max_seq_length)
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score, edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)
        else:
            inp_example = InputExample(texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
                                       label=score)
        test_samples.append(inp_example)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

# Reload the trained model for evaluation
model = ExtendSentenceTransformer(model_save_path)

# Evaluate the model on the test dataset
test_evaluator(model, output_path=model_save_path)

print("Model evaluation on test set completed.")
