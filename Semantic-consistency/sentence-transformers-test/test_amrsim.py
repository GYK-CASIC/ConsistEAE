import json
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import ExtendSentenceTransformer, InputExample
from preprocess import generate_ref_edge
group_size = 3053 
from tqdm import tqdm

with open("Syntactic-consistency/ace05ep/event_entity_types.json", 'r') as arg_file:
    arg_data = json.load(arg_file)

few_shot = 5


def arg_identity(event_type):
    return arg_data.get(event_type, "")


with open(f"Syntactic-consistency/ace05ep/Pre-extracted-data.json", 'r') as file:
    data = json.load(file)

with open("Syntactic-consistency/ace05ep/test_data.json", 'r') as train_file:
    train_data = json.load(train_file)

model_path = "output/ct-debug-bert"
model = ExtendSentenceTransformer(model_path)

test_sts_dataset_path = '/Semantic-consistency/data/ace05ep-test-train-full.amr'
error_log_path = 'Semantic-consistency/data/error_log.txt'

def return_simscore(model, examples):
    sim_scores_with_indices = []
    skipped_sentences = 0 

    for idx,example in enumerate(examples):
        sentences1 = example.texts[0]
        sentences2 = example.texts[1]
        
        ref1_graphs_index = example.edge_index[0]
        ref1_graphs_type = example.edge_type[0]
        ref1_pos_ids = example.pos_ids[0]
        
        max_seq_length = model.max_seq_length 
        if len(sentences1.split()) > max_seq_length or len(sentences2.split()) > max_seq_length:
            with open(error_log_path, 'a') as error_file:
                error_file.write(f"Warning: Sentences exceed max length ({max_seq_length} tokens)\n")
                error_file.write(f"example: {example}\n")
                error_file.write("-" * 50 + "\n")
            sim_scores_with_indices.append({
                'index': idx,
                'similarity_score': float('-inf'),
                'sentence1':sentences1,
                'sentence2':sentences2
            })
            skipped_sentences += 1
            continue 
        else:
            embeddings1 = model.encode([sentences1], graph_index=[ref1_graphs_index],
                                       graph_type=[ref1_graphs_type], batch_size=1,
                                       convert_to_numpy=True, pos_ids=[ref1_pos_ids])

            ref2_graphs_index = example.edge_index[1]
            ref2_graphs_type = example.edge_type[1]
            ref2_pos_ids = example.pos_ids[1]

            embeddings2 = model.encode([sentences2], graph_index=[ref2_graphs_index],
                                       graph_type=[ref2_graphs_type], batch_size=1,
                                       convert_to_numpy=True, pos_ids=[ref2_pos_ids])

            cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
            sim_scores_with_indices.append({
                'index': idx,
                'similarity_score': cosine_scores[0],
                'sentence1':sentences1,
                'sentence2':sentences2
            })
    return sim_scores_with_indices


test_samples = []
with open(test_sts_dataset_path, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="Processing test samples", unit="sample"):
        line = json.loads(line)
        triples_data = line['graph_ref2'].get('triples', "")
        max_seq_length = model.max_seq_length
        if not triples_data:
            print(f"Warning: 'triples' data is empty for this sample. Using default empty list.")
            line['graph_ref2']['triples'] = "[]" 
        edge_index, edge_type, pos_ids = generate_ref_edge(line, model.tokenizer, max_seq_length)
        inp_example = InputExample(
            texts=[line['graph_ref1']['amr_simple'], line['graph_ref2']['amr_simple']],
            edge_index=edge_index,
            edge_type=edge_type,
            pos_ids=pos_ids
        )
        test_samples.append(inp_example)

similar_results = return_simscore(model, test_samples)

prompt_list = []
sentence_list = []
for i in range(0,len(similar_results),group_size):
    print('i',i)
    group = similar_results[i:i+group_size]
    i = i//group_size
    sorted_group = sorted(group, key=lambda x: x['similarity_score'], reverse=True)
    top_5_matches = sorted_group[:5] 
    idx_in_group = [match['index'] for match in top_5_matches]    
    idx_in_group = [idx - i * group_size for idx in idx_in_group]
    train_item = train_data[i]

    sentence = train_item["sentence"]

    for event in train_item["event_mentions"]:
        event_type = event["event_type"]
        trigger = event["trigger"]["text"]
        arg = arg_identity(event_type)
        sentence_list.append(sentence)
        
        most_similar_indices = idx_in_group
        print(f"idx_in_group after adjustment: {idx_in_group}")
        examples = []
        for idx in most_similar_indices:
            
            print(f"Length of data: {len(data)}")
            most_similar_example = data[idx]
            for e_event in most_similar_example["event_mentions"]:
            
                e_trigger = e_event["trigger"]["text"]
                e_sentence = most_similar_example["sentence"]
                e_arg_dict = [{e_arg["role"]: e_arg["text"]} for e_arg in e_event["arguments"]]
                example_answer = json.dumps(e_arg_dict)
                example = f"""Example:
Given the following news about “{e_event['event_type']}” with the trigger “{e_trigger}”:
{e_sentence}
What is the “{arg}” for the “{e_event['event_type']}”? Please follow the format: {{arg_role:arg_text}}.
If there is no corresponding arg_role, please return Unknown. Do not return redundant text.
Example Answer: {example_answer}
"""
                examples.append(example)
                if len(examples) >= few_shot:
                    break

        if not examples:
            print("None")
            continue

        examples_text = "\n".join(examples)
        prompt = f"""Given the following news about “{event_type}” with the trigger “{trigger}”:
{sentence}
What is the “{arg}” for the “{event_type}”? Please follow the format: {{arg_role:arg_text}}.
If there is no corresponding arg_role, please return Unknown. Do not return redundant text.

{examples_text}

Now, analyze the given news and provide the arguments in the specified format.
"""
        prompt_list.append(prompt)


if prompt_list:
    print(prompt_list[0])

output_prompt_path = f"Semantic-consistency/data/ace05ep/prompt/prompt_EEA_with_trigger.json"
with open(output_prompt_path, "w", encoding="utf-8") as f:
    json.dump(prompt_list, f, indent=4, ensure_ascii=False)

print(f"Prompts saved to {output_prompt_path}")

def save_sim_scores_to_file(sim_scores_with_indices, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in sim_scores_with_indices:
            idx = item['index']
            score = item['similarity_score']
            sentence1 = item['sentence1']
            sentence2 = item['sentence2']
            f.write(f"{score}\n")

output_sim_scores_path = 'Semantic-consistency/ace05ep/similarity_scores.txt'

save_sim_scores_to_file(similar_results, output_sim_scores_path)

print(f"Similarity scores saved to {output_sim_scores_path}")
