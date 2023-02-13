import torch

from datasets import load_dataset
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

def run(testdata):
    datasets = load_dataset('smilegate-ai/kor_unsmile')
    model_name = 'smilegate-ai/kor_unsmile'
    model = BertForSequenceClassification.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = TextClassificationPipeline(
        model = model,
        tokenizer = tokenizer,
        device = 0,   # cpu: -1, gpu: gpu number
        # return_all_scores = True,
        top_k = 2,
        return_all_scores = False,
        function_to_apply = 'sigmoid'
    )
    for data in testdata:
        print(f"data result : \"{data}\"")
        for result in pipe(data)[0]:
            print(result)
        print()
            

if __name__ == "__main__":
    example_testcase = ["옛말에 암탉이 울면 나라 망한다고 했다", "꼭 키 작은 급식충이 이런 글 씀", "혐오가 당연한 건 없습니다"]
    run(example_testcase)


