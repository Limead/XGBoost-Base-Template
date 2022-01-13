# XGBoost-Base-Template
# 개요

프로젝트 수행 시 기본 템플릿의 형식 구성에 도움이 되고자 XGBoost + Bayesian optimization regression 모델링 과정에 대한 기초 템플릿을 구성해 보았습니다.

여기에 추가적으로 필요에 맞는 모듈을 덧붙여 나가면서 실제 업무에 도움이 되면 좋겠습니다

## 상세 내용 설명

- **내부 구조**
    - **XGB_main.py** : 템플릿을 실행하는 메인 py 스크립트 파일
    - **data** : data 셋을 저장해둔 폴더
        - 내부엔 전처리가 끝났다고 가정한 train_data.csv , test_data.csv 2개의 데이터를 저장합니다. ( 현재 예제에서는 최소한의 테스트를 위해 임의로 작성한 데이터로 저장되어 있습니다.)
    - **models** : model 학습을 위한 py 스크립트 파일을 정리해둔 폴더
        - XgbInput.py : XGBoost 모델링 수행 전 데이터셋을 train, valid set으로 나누고 target과 나머지 변수를 분리하는 등의 기능을 수행하는 스크립트 파일입니다.
        - XgbTrain.py : XGBoost 모델 메인 학습 구조를 설정한 스크립트 파일입니다.
        - XgbBayes.py : XGBoost 모델 HyperParameter 최적화를 위한 Bayesian optimization을 수행하는 스크립트 파일입니다.
    - **utils** : 필요에 따라 자주 사용하는 모듈을 정리해둔 폴더
        - scorer : 모델 학습 시 기준이 되는 score 값들을 정의한 스크립트 파일입니다 현재는 RMSE 1가지의 값만 정의되어 있습니다.
- **구조 설명**
    - main 파일에서 argument를 받아서 대부분의 값을 컨트롤 할 수 있도록 구성되어 있습니다.
    - 학습 과정은 데이터 클리닝(XgbInput.py) → 모델구조 정의(XgbTrain.py) → 하이퍼파라미터 튜닝 및 모델 학습(XgbBayes.py) 순으로 진행되게 됩니다.
- **argument 설명**
    - **데이터 경로**
        - —data_path : train_data의 경로를 설정합니다.
        - —test_path : test_data의 경로를 설정합니다.
    - **Target setting**
        - -target : 예측할 y 변수명을 설정합니다.
    - **Bayesian optimization setting**
        - -init_points : bayes opt 실행 시 처음 탐색할 parameter 조합의 수를 설정합니다.
        - -n_iter : 처음 init_points가 주어진 상태에서 몇번의 bayes 탐색을 진행할 지 설정합니다.
    - **model parameters**
        - XGBoost Model Parameter에 해당하는 값을 설정합니다. 설정값을 기준으로 일정 범위 내의 구간을 Bayes opt로 탐색하여 최종 parameter를 결정하게 됩니다.
            - n_estimators
            - learning_rate
            - max_depth
            - max_leaves
            - max_bin
            - min_child_weight
            - subsample
            - colsample_bytree
        - 자세한 parameter 설명은 아래 링크를 참조
        - [https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html)
- **사용 방법**
    - 전처리 된 데이터를 준비합니다
    - terminal에서 "python XGB_main.py -target {변수명} —data_path {train_data경로} —test_path {test_data경로}"
    - 위 명령어를 사용하여 간단하게 실행시킬 수 있습니다. 추가적으로 다른 argument를 설정하여 세부 값 조정이 가능합니다.
