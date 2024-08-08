# 본 코드는 모두 https://nbviewer.org/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/05_data_parallelism.ipynb 를 참고하여 작성했습니다.
# 분산처리 공부에 양질의 자료를 남겨주신 고현웅님께 정말 큰 감사 드립니다.

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

datasets = load_dataset("multi_nli").data["train"]
datasets = [
    {
        "premise": str(p),
        "hypothesis": str(h),
        "labels": l.as_py(),
    }
    for p, h, l in zip(datasets[2], datasets[5], datasets[9])
]
data_loader = DataLoader(datasets, batch_size=128, num_workers=4)

model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()

model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7], output_device=0)
optimizer = Adam(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss(reduction="mean")

for i, data in enumerate(data_loader):
    optimizer.zero_grad()
    tokens = tokenizer(
        data["premise"],
        data["hypothesis"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    model_input = {
        "input_ids": tokens.input_ids.cuda(),
        "attention_mask": tokens.attention_mask.cuda(),
        "return_dict": False
    }

    logits = model(**model_input)[0]

    loss = loss_fn(logits, data["labels"].cuda())
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"step: {i}, loss: {loss}")

    if i == 300:
        break


