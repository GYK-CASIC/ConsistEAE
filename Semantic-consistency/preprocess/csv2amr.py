import csv
import json

# 读取CSV数据
csv_file_path = '/data01/guoyikai/AMR/AMRSim-main/data/stsbenchmark/sts-dev.csv'
json_data = []

with open(csv_file_path, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t')
    
    for row in csv_reader:
        if len(row) >= 7:
            score = float(row[4])
            ref1 = row[5]
            ref2 = row[6]
            
            # 构建JSON格式数据
            data_entry = {
                'score': score,
                'ref1': ref1,
                'ref2': ref2
            }
            
            json_data.append(data_entry)

# 将JSON数据写入文件
json_file_path = '/data01/guoyikai/AMR/AMRSim-main/data/stsbenchmark/sts-dev.json'
with open(json_file_path, 'w') as jsonfile:
    json.dump(json_data, jsonfile, indent=2)

print(f'Data has been processed and saved to {json_file_path}.')
