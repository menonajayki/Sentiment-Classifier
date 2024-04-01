from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def text_generator(text):
    encode_text = tokenizer.encode(text, return_tensors='pt')
    print (encode_text)

text_generated = text_generator("Hello my world")
print(text_generated)