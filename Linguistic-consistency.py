#!/usr/bin/env python3

import json
import math
import stanza
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

# Create a lock object
cache_lock = threading.Lock()

# Download and load the English model
model_dir = "Syntactic-consistency/stanza_resource"
nlp = stanza.Pipeline('en', dir=model_dir, download_method=None)

# Read the event entity type file
with open("Syntactic-consistency/ace05ep/event_entity_types.json", 'r') as arg_file:
    arg_data = json.load(arg_file)

few_shot = 5


def arg_identity(event_type):
    """Get parameters corresponding to the event type"""
    return arg_data.get(event_type, "")


# Load example dataset
with open(f"Syntactic-consistency/ace05e/pre-extracted-data.json", 'r') as file:
    data = json.load(file)

# Load test data
with open("Syntactic-consistency/ace05ep/test_data.json", 'r') as train_file:
    train_data = json.load(train_file)


def load_similarity_scores_from_file(file_path):
    """
    Read similarity scores from a file, assuming each line contains a floating-point number.
    :param file_path: File path storing similarity scores
    :return: A list containing all similarity scores
    """
    with open(file_path, 'r') as f:
        scores = [float(line.strip()) for line in f.readlines()]
        # print(f"Loaded similarity scores: {scores}")  # Print similarity scores
    return scores


# Extract dependency relations from the syntactic tree
def extract_dep_tree(doc):
    dep_tree = []
    for sentence in doc.sentences:
        for word in sentence.words:
            dep_tree.append((word.text, word.deprel, word.head))
    return dep_tree


# Convert dependency tree to a NetworkX graph
def build_tree_graph(dep_tree):
    G = nx.DiGraph()
    for word, dep_rel, head in dep_tree:
        if head != 0:  # Exclude root node
            parent_word = dep_tree[head - 1][0]  # head is 1-based index
            G.add_edge(parent_word, word, rel=dep_rel)  # Edge from parent node to child node
    return G


# Tree edit distance calculation (Dynamic Programming)
def tree_edit_distance(G1, G2):
    '''
    Input: Two trees
    Output: Tree edit distance
    :param G1:
    :param G2:
    :return: Tree edit distance between G1 and G2 (an integer)
    '''
    nodes1 = list(G1.nodes())
    nodes2 = list(G2.nodes())
    dp = [[0] * (len(nodes2) + 1) for _ in range(len(nodes1) + 1)]

    for i in range(len(nodes1) + 1):
        dp[i][0] = i  # Delete all nodes from G1
    for j in range(len(nodes2) + 1):
        dp[0][j] = j  # Insert all nodes into G1

    for i in range(1, len(nodes1) + 1):
        for j in range(1, len(nodes2) + 1):
            cost = 0 if nodes1[i - 1] == nodes2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost  # Replacement
            )

    return 1 / (1 + math.log(dp[len(nodes1)][len(nodes2)]))


# Get sentence structure similarity
def get_sentence_structure_similarity(sentence1, sentence2):
    """
    Get structural similarity between the target sentence and the example sentence, using cache
    :param sentence1: Target sentence
    :param sentence2: Example sentence
    :return: Similarity score between the target and example sentences
    """
    # Cache lookup

    # print(f"Cache miss for: {sentence1} <-> {sentence2}")
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    dep_tree1 = extract_dep_tree(doc1)
    dep_tree2 = extract_dep_tree(doc2)

    graph1 = build_tree_graph(dep_tree1)
    graph2 = build_tree_graph(dep_tree2)

    similarity_score = tree_edit_distance(graph1, graph2)

    return similarity_score


def find_most_similar_examples_by_structure_random(target_sentence, target_event_type, example_sentences,
                                                   top_n=few_shot, similarity_cache=None, similarity_file_path="Syntactic-consistency/ace05ep/similarity_scores.txt", i=None):
    """
    Retrieve the top `top_n` most similar example sentences to the target sentence
    :param target_sentence: Target sentence
    :param target_event_type: Event type
    :param example_sentences: Example sentences
    :param top_n: Number of most similar sentences to retrieve
    :param similarity_file_path: Path to similarity score file
    :param i: Current index being processed
    :return: List of indexes of the most similar example sentences
    """
    # Read similarity scores from file
    if similarity_file_path:
        file_scores = load_similarity_scores_from_file(similarity_file_path)
    else:
        file_scores = []

    # Ensure `i` is provided and valid
    if i is None or i <= 0:
        print(i)
        raise ValueError("i must be greater than 0, representing the current data entry being processed")

    # Compute similarity score range
    start_idx = (i - 1) * 50
    end_idx = i * 50  # Exclusive end_idx

    # Retrieve similarity scores for the current processing data
    current_file_scores = file_scores[start_idx:end_idx]

    filtered_examples = [
        (idx, example) for idx, example in enumerate(data)
        if any(e_event["event_type"] == target_event_type for e_event in example["event_mentions"])
    ]

    if not filtered_examples:
        print(f"No examples with the same event type, using file similarity for ranking (Range: {start_idx}-{end_idx})")

        # When filtered_examples is empty, sort based on file similarity and select top `top_n`
        combined_similarities = [(idx + start_idx, file_score) for idx, file_score in enumerate(current_file_scores)]

        # Sort by similarity in descending order
        combined_similarities.sort(key=lambda x: x[1], reverse=True)

        # Get the indexes of the top `top_n` most similar sentences
        most_similar_examples = [combined_similarities[idx][0] - 50 * (i - 1) for idx in range(min(top_n, len(combined_similarities)))]

        return most_similar_examples

    # If related examples exist, continue computing structural similarity
    similarities = []

    # Use thread pool for parallel similarity computation
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(get_sentence_structure_similarity, target_sentence, example["sentence"]): (idx, example["sentence"])
            for idx, example in filtered_examples
        }

        # Show progress bar with tqdm
        for future in tqdm(futures, desc="Calculating similarities", total=len(futures)):
            similarity_score = future.result()
            similarities.append((futures[future][0], similarity_score))  # Save sentence index and similarity score

    # Combine structure and semantic similarity
    combined_similarities = []
    for idx, score in similarities:
        print("Structurally similar ID:", idx)
        file_score = current_file_scores[idx]  # Use current file similarity
        combined_score = 2 * score + 1 * file_score
        combined_similarities.append((idx, combined_score))

    # Sort by similarity in descending order
    combined_similarities.sort(key=lambda x: x[1], reverse=True)

    # Get top `top_n` most similar sentences
    most_similar_examples = [filtered_examples[idx][0] for idx in range(min(top_n, len(similarities)))]
    return most_similar_examples

# Prepare a list to store prompts
prompt_list = []
sentence_list = []

for train_item in train_data:
    print("Currently processing sentence:", train_item["sentence"])
    if not train_item.get("event_mentions"):
        continue
    sentence = train_item["sentence"]

    for event in train_item["event_mentions"]:
        event_type = event["event_type"]
        trigger = event["trigger"]["text"]
        arg = arg_identity(event_type)
        sentence_list.append(sentence)

        # Select the most similar examples
        most_similar_examples = find_most_similar_examples_by_structure_random(sentence, event_type, data)
        examples = []
        for idx in most_similar_examples:
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

# Print the first prompt for verification
if prompt_list:
    print(prompt_list[0])

# Save prompts and sentences
output_prompt_path = f"Syntactic-consistency/ace05ep/prompt_EEA_with_trigger_merge.json"
with open(output_prompt_path, "w", encoding="utf-8") as f:
    json.dump(prompt_list, f, indent=4, ensure_ascii=False)

print(f"Prompts saved to {output_prompt_path}")
