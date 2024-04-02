from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def text_generator(text):
    encode_text = tokenizer.encode(text, return_tensors='pt')
    print("Encoded Text:")
    print(encode_text)
    next_sentences = model.generate(encode_text)
    decode_text = tokenizer.decode(next_sentences[0], skip_special_tokens=True)
    print("Generated Text:")
    print(decode_text)


text_generated = text_generator("best friends")
