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
학습부터 소켓통신까지 모든 기능이 다 들어있는 코드입니다.  
```
분석+학습+학습모델 저장+소켓통신.py
```

## 2. 실행
코드에 필요한 라이브러리를 설치해줍니다.  
```
Okt
konlpy
pandas
```
코드와 같은 폴더에 txt파일을 두고 코드를 실행시켜줍니다.  
```
python 분석+학습+학습모델 저장+소켓통신.py
```
아래 코드 부분에서 학습횟수와 batch_size를 조정해 줍니다.  
```
model.fit(x_train, y_train, epochs=30, batch_size=512)
```
학습이 완료되면 아래에 지정된 주소로 소켓통신이 생성됩니다.  
이하 모델 로딩+소켓통신.py도 동일  
```
HOST = ''
PORT = 12345
```
통신으로 문장을 받아오면 모델에 넣고 결과값을 받아 출력  
```
if (score > 0.5):
        print(f"{review} ==> 긍정 ({round(score * 100)}%)")
        emotion = "긍정"
        return emotion
    else:
        print(f"{review} ==> 부정 ({round((1 - score) * 100)}%)")
        emotion = "부정"
        return emotion
```
