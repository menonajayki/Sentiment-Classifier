from transformers import GPT2Tokenizer, GPT2LMHeadModel


pad_token = 'pad_token'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', padding_side='left')
tokenizer.add_special_tokens({'pad_token': pad_token})
model = GPT2LMHeadModel.from_pretrained('gpt2-xl', pad_token_id=tokenizer.pad_token_id)


def text_generator(input_texts):
    generated_texts = []

    for input_text in input_texts:
        input_text_chunks = [input_text[i:i + 512] for i in range(0, len(input_text), 512)]
        generated_text_chunks = []

        for chunk in input_text_chunks:
            input_ids = tokenizer.encode(chunk, return_tensors='pt')

            outputs = model.generate(input_ids, max_length=1024, num_return_sequences=1, no_repeat_ngram_size=2,
                                     early_stopping=True, num_beams=5)

            generated_text_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text_chunks.append(generated_text_chunk)

        generated_text = " ".join(generated_text_chunks)
        generated_texts.append(generated_text)

    return generated_texts

input_texts = [
    "Hello my world.", "This is fun."
]


generated_texts = text_generator(input_texts)

for i, text in enumerate(generated_texts):
    print(f"Generated Text {i + 1}:")
    print(text)
