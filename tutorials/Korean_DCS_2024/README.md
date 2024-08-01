# 일상 대화 요약 Baseline
본 리포지토리는 '2024년 국립국어원 인공지능의 한국어 능력 평가' 경진 대회 과제 중 '일상 대화 요약'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.  

학습, 추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.   

|Model|Evaluation Score|ROUGE-1|bertscore|bluert|
|:---|---|---|---|---|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (without SFT)|54.276|44.592|73.277|44.958|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (with SFT)|58.982|54.759|79.154|43.035|

## 리포지토리 구조 (Repository Structure)
```
# 학습에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
├── test.py
└── train.py

# 학습에 사용될 커스텀 함수들을 보관하는 디렉토리
src
├── data.py     # Custom Dataset
└── utils.py
```

## 데이터 (Data)
```
{
    "id": "nikluge-2024-일상 대화 요약-train-000001",
    "input": {
        "conversation": [
            {
                "speaker": "SD2000001",
                "utterance": "저는 여행 다니는 것을 굉장히 좋아하는데요. 그래가지고 스페인이나 뭐 영국 유럽 아니면 국내에서도 뭐 강릉이나 전주 같은 데를 많이 다녔는데"
            },
            {
                "speaker": "SD2000001",
                "utterance": "혹시 여행 다니는 거 좋아하시나요?"
            },
            {
                "speaker": "SD2000002",
                "utterance": "저 여행 다니는 거 되게 좋아해서 대학교 내내 여행을 엄청 많이 다녔었는데요."
            },
            ...
            ...
            ...
        ],
        "subject_keyword": [
            "해외여행"
        ]
    },
    "output": "이 대화에서 화자들은 좋았던 여행지와 기억나는 주요 명소에 대해 이야기했습니다. SD2000001은 여행을 좋아하여 국내, 해외 여행을 많이 다녔다고 말했습니다. 특히 기억에 남는 여행지로 스페인 마드리드의 톨레도를 소개했습니다. 그 중 화려하게 꾸며진 대성당과 디저트가 인상적이었다고 이야기했습니다. SD2000002는 대학교에 진학한 후 해외여행을 자주 다녔고, 스페인과 포루투갈이 가장 기억에 남는 여행지라고 말했습니다. 그리고 톨레도도 다녀왔지만 날씨가 더워서 제대로 구경하지 못했다는 경험을 이야기했습니다."
}
```

## 실행 방법 (How to Run)
### 학습 (Train)
```
CUDA_VISIBLE_DEVICES=1,3 python -m run.train \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --epoch 5 \
    --lr 2e-5 \
    --warmup_steps 20
```

### 추론 (Inference)
```
python -m run.test \
    --output result.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0
```

## Reference

huggingface/transformers (https://github.com/huggingface/transformers)  
Bllossome (Teddysum) (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
