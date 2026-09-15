from jsonlines import jsonlines


def evaluate_accuracy(input_file: str):
    model_correct = {"bggpt-gemma2": 0}
    model_total = {"bggpt-gemma2": 0}

    with jsonlines.open(input_file) as infile:
        for response in infile:
            correct_answer = response['correct_answer']
            for model, answer in response['model_answers'].items():
                model_total[model] += 1
                if answer == correct_answer:
                    model_correct[model] += 1
                else:
                    print(f"Q: {response['question']}, Correct: {correct_answer}, BgGPT: {answer}")

    for model in model_correct.keys():
        accuracy = (model_correct[model] / model_total[model]) * 100 if model_total[model] else 0
        print(f"{model} Accuracy: {accuracy:.2f}%")


def evaluate_accuracy_kg_rag(input_file: str):
    model_correct = 0
    model_total = 0

    with jsonlines.open(input_file) as infile:
        for response in infile:
            correct_answer = response['correct_answer']
            model_total += 1
            if response['bggpt_kg_rag_answer'] == correct_answer:
                model_correct += 1
            else:
                print(f"Q: {response['question']}, Correct: {correct_answer}, BgGPT: {response['bggpt_kg_rag_answer']}")

    accuracy = (model_correct / model_total) * 100
    print(f"BgGPT KG RAG Accuracy: {accuracy:.2f}%")


def evaluate_accuracy_vector_rag(input_file: str):
    model_correct = 0
    model_total = 0

    with jsonlines.open(input_file) as infile:
        for response in infile:
            correct_answer = response['correct_answer']
            model_total += 1
            if response['bggpt_vector_rag_answer'] == correct_answer:
                model_correct += 1

    accuracy = (model_correct / model_total) * 100
    print(f"BgGPT Vector RAG Accuracy Not Tuned Retrieval: {accuracy:.2f}%")


if __name__ == "__main__":
    input_file = 'testset/zeroshot_kg_rag_bggpt_curated_exams.jsonl'
    evaluate_accuracy(input_file)
    evaluate_accuracy_kg_rag(input_file)
    # evaluate_accuracy_vector_rag(input_file)
