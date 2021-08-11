# use-KONLPY-positive-negative-analyzer
KONLPY를 이용한 긍부정 분석기 인공지능

## 1. 설명
### 구조
```
   use-KONLPY-positive-negative-analyzer
     |- ratings_test.txt
     |- ratings_train.txt
     |- 모델 로딩+소켓통신.py
     |- 분석+학습+학습모델 저장+소켓통신.py
```
네이버 영화리뷰 데이터를 이용한 긍정과 부정을 학습할 수 있는 데이터 입니다.  
```
ratings_train.txt  
ratings_test.txt  
```
학습된 모델이 있다면 불러와서 바로 통신에 사용할 수 있는 코드입니다.  
```
모델 로딩+소켓통신.py
```
