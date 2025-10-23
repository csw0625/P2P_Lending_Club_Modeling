# P2P_Loan_Modeling
### SNU KDT 6조
### 권서영 이소민 이유림 조성우 지희선 홍예나
> **목표**: Lending Club(P2P) 대출 데이터와 미국 국채 수익률을 결합해 **초과수익 여부**를 분류하여 포트폴리오 **Sharpe Ratio**가 최대가 되는 Threshold 선정  
> **프로세스**: Data Prep → **IRR 계산 & 라벨링**(대출 IRR vs 국채수익) → **모델링** → **Threshold 탐색**(0.01–0.99) → **Sharpe 평가**(부트스트랩) → **최종 모델 선정**


<pre>
.
├── README.md
├── 보고서.pdf
├── 발표자료.pdf
├── Final_Code(Preprocessing, XGBoost).ipynb
├── preprocessing.ipynb
├── data.zip                         # Datasets
│   ├── GS3.csv                      # 36개월 만기 미국 국채
│   ├── GS5.csv                      # 60개월 만기 미국 국채
│   ├── lending_club_2020_train.csv  # Train Set
│   └── lending_club_2020_test.csv   # Test Set
├── models/
│   └── lending_club_model.pkl       # Final XGBoost
└── notebooks/
    ├── Naive_Bayes.ipynb
    ├── SVC.ipynb
    ├── Random_Forest.ipynb
    ├── LightGBM.ipynb
    └── MLP.ipynb
</pre>

--- 

## 1. 프로젝트 개요
  - Lending Club 2020 대출 데이터 + 미국 국채 수익률 결합
  - 각 대출의 연간 IRR과 동일 만기 국채 수익률을 비교해 초과수익 여부를 라벨로 정의
  - 분류 확률의 임곗값을 탐색해 포트폴리오 **Sharpe Ratio** 최대화
  - 국채 매핑: **GS3 = 36개월 만기**, **GS5 = 60개월 만기**
  - 최종 선택: **XGBoost**, 최적 임곗값 0.3195, 최고 Sharpe 0.2016
---




## 2. 분석

### 2.1 모델 평가 방법
- 목표 지표: **Sharpe Ratio** = 초과수익률 평균 ÷ 초과수익률 표준편차
- 라벨 정의
  - `y = 1` : 대출 Annual IRR < 대출 발생 시점 동일 만기 국채 수익률 → 포트폴리오 추가
  - `y = 0` : 대출 Annual IRR ≥ 대출 발생 시점 동일 만기 국채 수익률 → 포트폴리오 투자
- 평가 절차
  1) 분류기에서 샘플별 확률 산출  
  2) 임곗값 0.01~0.99 스윕  
  3) 임곗값별 투자/제외로 구성된 포트폴리오 초과수익률 계산  
  4) Sharpe Ratio 최대화 임곗값 선택(부트스트랩으로 평균·표준편차 확인)

### 2.2 분석 Flow

**Preprocessing**
- 결측 관리
  - 결측비율 ≥ 40% 열 제거
  - 결측 패턴 상관 높은 변수 조합은 보간 예측자에서 제외
- 보간
  - Custom IterativeImputer로 수렴·편향 제어
- 수치 변환
  - Quantile Transformation으로 분포 안정화
- 범주 인코딩(차원 최소화)
  - `emp_length` ordinal encoding
  - `home_ownership` 주요 3개만 one-hot(MORTGAGE/OWN/RENT), 기타 통합
  - `purpose` 4그룹 재범주화 후 주요 그룹만 one-hot
  - `verification_status`, `initial_list_status` binary encoding
- 기타 처리
  - `revol_util`의 % 제거, `fico_range_high/low` 평균으로 `avg_fico` 생성

**Modeling**
- Naive Bayes, SVC(RBF), Random Forest, LightGBM, **XGBoost**, MLP
- 공통 절차: 확률 산출 → 임곗값 스윕 → Sharpe 계산 → 부트스트랩 안정성 확인

**최종모델 선정**
- XGBoost 선택
  - 최고 Sharpe 0.2016
  - 최적 임곗값 0.3195

### 2.3 SHAP
- 특징 영향 요약
  - 긍정적: 36개월 단기, 높은 신용지표(avg_fico), 높은 연소득, 낮은 DTI
  - 부정적: 60개월 장기, 대출금액 증가, 높은 DTI
- 패턴
  - 단기·소액·저부채 특성이 초과수익 가능성에 유리

---

## 3. 결론
- 포트폴리오 수익률 기반 라벨링과 임곗값 최적화로 투자 의사결정 중심의 분류 파이프라인 구성
- **XGBoost + 임곗값 0.3195**가 최고 Sharpe 0.2016 기록
- 시사점
  - 임곗값 탐색을 통해 포트폴리오 관점 성과를 직접 최적화
  - 단기·저위험 특성 편입이 Sharpe 개선에 기여
- 한계 및 향후 과제
  - High Cardinality 범주형 변수 추가 활용(직업)
  - 불균형 대응(비용민감 학습, calibration 계열) 고도화
  - Sharpe 직접 목적함수 연구(전역 지표의 안정적 근사)
