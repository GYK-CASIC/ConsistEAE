import json
import re

def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def remove_null_values(data):
    return {k: v for k, v in data.items() if v is not None}

def extract_json_from_content(content):

    if content is None:
        return {}

    json_match = re.search(r'{.*}', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {} 

    return {} 
def transform_data(data):
    transformed_data = []

    for entry in data:
        print(entry)
        sentence = entry['sentence']

        content = extract_json_from_content(entry['content'])


        content = remove_null_values(content)

        arguments = []
        for role, items in content.items():
            if items: 
                if isinstance(items, list):  
                    for item in items:
                        arguments.append({"text": item, "role": role})
                else:  
                    arguments.append({"text": items, "role": role})

        transformed_data.append({
            "sentence": sentence,
            "arguments": arguments
        })

    return transformed_data



def write_output_file(output_data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)


def add_event_info_to_response(filter_data, response_data):
    for i in range(len(filter_data)):
        for event in filter_data[i]["event_mentions"]:
            event_type = event["event_type"]
            response_data[i]["event_type"] = event_type
            response_data[i]["trigger"] = event["trigger"]
            response_data[i]["sentence"] = filter_data[i]["sentence"]
    return response_data

def merge_event_mentions(new_data):
    event_mentions_corrected_data = {}

    for item in new_data:
        sentence = item['sentence']
        new_event_type = item.get('event_type', None)
        new_trigger = item.get('trigger', None)
        new_arguments = item.get('arguments', [])

        new_event_mention = {
            'event_type': new_event_type,
            'trigger': new_trigger,
            'arguments': new_arguments
        }

        if sentence in event_mentions_corrected_data:
            event_mentions_corrected_data[sentence]['event_mentions'].append(new_event_mention)

    final_event_mentions_list = list(event_mentions_corrected_data.values())
    return final_event_mentions_list

def main(input_file_path, filter_file_path, output_file_path):
    data = read_input_file(input_file_path)
    filter_data = read_input_file(filter_file_path)

    transformed_data = transform_data(data)

    response_data = transformed_data
    write_output_file(response_data, output_file_path)


input_file = 'EAE/response.json' 
filter_file = '' 
output_file = 'EAE/response.json' 


main(input_file, filter_file, output_file)
