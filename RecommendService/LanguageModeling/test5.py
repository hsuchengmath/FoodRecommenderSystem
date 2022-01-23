


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('AlbertHSU/ChineseFoodBert')
model = AutoModel.from_pretrained('AlbertHSU/ChineseFoodBert')

inputs = tokenizer("碳燒烤鴨", return_tensors="pt")
outputs = model(**inputs)
print(outputs) # torch.Size([1, 768])