# 본 코드는 모두 https://nbviewer.org/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/05_data_parallelism.ipynb 를 참고하여 작성했습니다.
# 분산처리 공부에 양질의 자료를 남겨주신 고현웅님께 정말 큰 감사 드립니다.

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()

datasets = load_dataset("multi_nli").data["train"]
datasets = [
    {
        "premise": str(p),
        "hypothesis": str(h),
        "labels": l.as_py(),
    }
    for p, h, l in zip(datasets[2], datasets[5], datasets[9])
]

sampler = DistributedSampler(
    datasets,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)
data_loader = DataLoader(
    datasets,
    batch_size=32,
    num_workers=4,
    sampler=sampler,
    shuffle=False,
    pin_memory=True,
)

model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()
model = DistributedDataParallel(model, device_ids=[device], output_device=device)
optimizer = Adam(model.parameters(), lr=3e-5)

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

    loss = model(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        labels=data["labels"],
    ).loss

    loss.backward()
    optimizer.step()

    if i % 10 == 0 and rank == 0:
        print(f"step:{i}, loss:{loss}")

    if i == 300:
        break