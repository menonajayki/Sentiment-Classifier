from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

pad_token = 'pad_token'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
tokenizer.add_special_tokens({'pad_token': pad_token})
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.pad_token_id)

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


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.json
    input_text = data["input_text"]

    generated_texts = text_generator([input_text])

    response = {
        "generated_text": generated_texts[0]
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
