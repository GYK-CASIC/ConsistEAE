import json

def calculate_metrics(gt_data, pred_data):
    gt_num = 0  # Total number of arguments in ground truth data
    pred_num = 0  # Total number of arguments in predicted data
    arg_i = 0  # Correctly matched arguments (arg-i)
    arg_c = 0  # Correctly matched arguments with the same role (arg-c)

    # Count the number of arguments in ground truth data
    for item in gt_data:
        gt_num += len(item['arguments'])  # Count each argument

    # Count the number of arguments in predicted data and check for matches
    for i, pred_item in enumerate(pred_data):
        pred_sentence = pred_item['sentence']
        print(pred_sentence)
        pred_arguments = pred_item['arguments']
        pred_num += len(pred_arguments)

        # Retrieve the corresponding ground truth data item using the index
        matching_gt_item = gt_data[i]

        # Arguments in the ground truth data
        gt_arguments = matching_gt_item['arguments']

        # Iterate over each argument in predicted data and match with ground truth
        for pred_arg in pred_arguments:
            # **Match arg-i (Exact text match) or core_words match**
            matching_gt_arg_i = next((
                arg for arg in gt_arguments
                if arg['text'] == pred_arg['text'] or
                   any(core_word == arg['text'] for core_word in pred_arg.get('core_words', [])) or
                   # New: Directly match using set intersection with head_words
                   set(pred_arg.get('core_words', [])) & set(arg.get('core_words', []))
            ), None)
            if matching_gt_arg_i:
                arg_i += 1  # If a perfect argument match is found (text match or core_word match)

            # **Match arg-c (Exact text and role match) or core_words match**
            matching_gt_arg_c = next((
                arg for arg in gt_arguments
                if arg['role'] == pred_arg['role'] and (
                    # Check if the predicted argument's text or core_word matches the ground truth
                    arg['text'] == pred_arg['text'] or
                    any(core_word == arg['text']
                        for core_word in pred_arg.get('core_words', [])) or
                    set(pred_arg.get('head_words', [])) & set(arg.get('head_words', []))
            )
            ), None)
            if matching_gt_arg_c:
                arg_c += 1  # If a perfect match is found (text and role match or core_word match)

    # Calculate precision, recall, and F1-score for arg-i
    precision_arg_i = arg_i / pred_num if pred_num > 0 else 0
    recall_arg_i = arg_i / gt_num if gt_num > 0 else 0
    arg_i_f1 = 2 * (precision_arg_i * recall_arg_i) / (precision_arg_i + recall_arg_i) if (
                    precision_arg_i + recall_arg_i) > 0 else 0

    # Calculate precision, recall, and F1-score for arg-c
    precision_arg_c = arg_c / pred_num if pred_num > 0 else 0
    recall_arg_c = arg_c / gt_num if gt_num > 0 else 0
    arg_c_f1 = 2 * (precision_arg_c * recall_arg_c) / (precision_arg_c + recall_arg_c) if (
                    precision_arg_c + recall_arg_c) > 0 else 0

    return {
        'arg-i F1 Score': arg_i_f1,
        'arg-c F1 Score': arg_c_f1,
        'Correct Argument Matches (arg-i)': arg_i,
        'Correct Role and Text Matches (arg-c)': arg_c,
        'Total Ground Truth Arguments': gt_num,
        'Total Predicted Arguments': pred_num
    }

# Load data and execute
# train_data = json.load(open("ACE05E/data/ace05e/split_train_data.json", "r", encoding="utf-8"))
train_data = json.load(open("EAE/ACE2005ep.json", "r", encoding="utf-8"))
response_data = json.load(open("EAE/response.json", "r", encoding="utf-8"))
results = calculate_metrics(train_data, response_data)
print(results)
