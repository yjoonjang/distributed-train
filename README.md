# distributed-train
**distributed-train**은 분산 처리 기술을 활용하여 모델 학습의 효율성을 극대화하는 방법을 실습하는 과정입니다.

## 1. 데이터 분산 처리 (DP & DDP)
#### 관련 블로그: [DataParallel(DP) vs DistributedDataParallel(DDP)](https://medium.com/@yjoonjang/%EB%B6%84%EC%82%B0-%EC%B2%98%EB%A6%AC-1-dataparallel-dp-vs-distributeddataparallel-ddp-feat-python-gil-056324f90f3c)
- `1_dp_tutorial.py`: **DataParallel(DP)**을 사용한 분산 학습 실습.
- `2_ddp_tutorial.py`: **DistributedDataParallel(DDP)**을 사용한 고성능 분산 학습 실습.
- 이 두 실습을 통해 데이터 분산 처리 방식의 차이를 이해하고, 각 방식의 장단점을 파악합니다.

## 2. 모델 분산 처리 (PP, TP, MP)
#### 관련 블로그: [Pipeline Parallelism(PP)와 Tensor Parallelism(TP)](https://medium.com/@yjoonjang/%EB%B6%84%EC%82%B0-%EC%B2%98%EB%A6%AC-3-pipeline-parallelism%EA%B3%BC-tensor-parallelism%EC%97%90-%EA%B4%80%ED%95%98%EC%97%AC-7b4420fe0281)
- `4_mp_with_accelerate_tutorial.ipynb`: **Accelerate 라이브러리의 MP**을 사용한 분산 학습 실습.

## 3. Mixed Precision Training
#### 관련 블로그: [Mixed Precision Training](https://medium.com/@yjoonjang/mixed-precision-training%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-mp-amp-torch-cuda-amp-15c99488ed34)
- `3_mixed_precision_tutorial.py`: **Mixed Precision Training** 기법을 사용하여 학습 속도를 높이고 메모리 사용량을 줄이는 방법을 실습합니다.

## 4. Deepspeed (ZeRO)
#### 관련 블로그: [ZeRO-DP](https://medium.com/@yjoonjang/%EB%B6%84%EC%82%B0-%EC%B2%98%EB%A6%AC-4-zero-dp-%EC%99%84%EC%A0%84-%EC%9D%B4%ED%95%B4-feat-deepspeed-87b9c98a4d8b)
- `5_zero_dp_tutorial.py`: **ZeRO-DP** 기법을 활용하여 메모리 최적화된 분산 처리 방법을 실습합니다.
#### 말평 실습 with deepspeed
- teddysum에서 주최하는 [일상 대화 요약](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=146&clCd=END_TASK&subMenuId=sub01)의 베이스라인 모델인 [llama-3-Korean-Bllossom-8B](https://github.com/teddysum/Korean_DCS_2024)에 **Deepspeed**를 적용하여 모델을 fine-tuning 합니다.
- `bash scripts/finetune.sh`명령어를 통해 Deepspeed를 활용한 분산 학습을 실습합니다.
- 이 단계를 통해 Deepspeed의 ZeRO 최적화 기법을 활용한 대규모 모델 학습의 성능 향상을 직접 경험할 수 있습니다.

