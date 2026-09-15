import json

import jsonlines
import requests

from geograph.constants import BGGPT_API_KEY, BGGPT_MODEL, BGGPT_V2_API_URL


def generate_and_write_multiple_responses_bggpt(input_file: str, output_file: str):
    with jsonlines.open(input_file) as input_reader:
        with jsonlines.open(output_file, mode='w') as output_writer:
            for line in input_reader:
                data = generate_single_response_bggpt(line['prompt'])
                output_writer.write(data)


def generate_single_response_bggpt(prompt: str):
    inst_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    payload = {
        "max_tokens": 1000,
        "model": BGGPT_MODEL,
        "stop": ["<end_of_turn>", "<eos>"],
        "temperature": 0.0,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "stream": True,
        "prompt": inst_prompt
    }
    headers = {
        "apikey": BGGPT_API_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    response = requests.post(
        BGGPT_V2_API_URL,
        headers=headers,
        json=payload,
        stream=True
    )

    if response.status_code == 200:
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")

                        accumulated_response += content
                    except json.JSONDecodeError:
                        continue

        data = {
            'prompt': prompt,
            'response': accumulated_response.strip()
        }

        # print(accumulated_response.strip())
        return data
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


def create_prompt(question, options):
    prompt = f"""
Ще ти бъде даден въпрос от затворен тип с 4 опции: A, B, C и D. Отговори като използваш само буквата на верния отговор.

Въпрос: {question}
"""
    for option in options:
        prompt += f"{option['label']}) {option['text']}\n"
    prompt += "Отговор:"
    return prompt


def generate_response_bggpt_gemma2(prompt: str):
    inst_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    payload = {
        "max_tokens": 2,
        "model": BGGPT_MODEL,
        "stop": ["<end_of_turn>", "<eos>"],
        "temperature": 0.0,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "stream": True,
        "prompt": inst_prompt
    }
    headers = {
        "apikey": BGGPT_API_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    response = requests.post(
        BGGPT_V2_API_URL,
        headers=headers,
        json=payload,
        stream=True
    )
    is_debug = False
    if response.status_code == 200:
        if is_debug: # change max_tokens to 500
            full_content = []
            lines = response.content.decode('utf-8').strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("data: "):
                    json_str = line[len("data: "):].strip()
                    if json_str and json_str != "[DONE]":
                        try:
                            data_obj = json.loads(json_str)
                            if ("choices" in data_obj and len(data_obj["choices"]) > 0 and "delta" in data_obj["choices"][0] and "content" in data_obj["choices"][0]["delta"]):
                                content_chunk = data_obj["choices"][0]["delta"]["content"]
                                if content_chunk:
                                    full_content.append(content_chunk)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e} on line: {json_str}")
                elif line == "data: [DONE]":
                    break
            result = "".join(full_content)
            print(f"Parsed content: '{result}'")
        else:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("text", "")
                            if len(content) == 1 and content.isalpha():
                                return content
                        except json.JSONDecodeError:
                            continue
        return None
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
