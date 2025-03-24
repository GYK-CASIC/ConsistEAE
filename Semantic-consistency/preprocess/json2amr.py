from transformers import BartTokenizer, BartForConditionalGeneration

# 加载预训练的SPRING模型和分词器
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# 输入句子
sentence = "The boy wants to eat an apple."

# 编码并生成 AMR 表示
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)

# 解码并显示 AMR 表示
amr = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(amr)
