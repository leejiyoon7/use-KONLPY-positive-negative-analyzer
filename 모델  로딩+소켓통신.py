from konlpy.tag import Okt
import os
import json
import pprint

pp = pprint.PrettyPrinter(width=40, indent=4)
okt = Okt()


def tokenize(doc):
    # 형태소와 품사를 join
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json', 'rt', encoding="utf-8") as f:
        train_docs = json.load(f)
    with open('test_docs.json', 'rt', encoding="utf-8") as f:
        test_docs = json.load(f)

print("테스트 데이터 출력 (train_docs)")
pp.pprint(train_docs[0])
print("테스트 데이터 출력 (test_docs)")
pp.pprint(test_docs[0])

tokens = [t for d in train_docs for t in d[0]]

import nltk

text = nltk.Text(tokens, name='NMSC')

# 토큰개수
print("총 토큰의 개수")
print(len(text.tokens))

# 중복을 제외한 토큰개수
print("중복을 제외한 토큰의 개수")
print(len(set(text.tokens)))

# 출력빈도가 높은 상위 토큰 10개
print("출력빈도가 높은 상위 토큰 10개")
pp.pprint(text.vocab().most_common(10))

FREQUENCY_COUNT = 10000
if os.path.isfile('selected_words.json'):
    with open('selected_words.json', 'rt', encoding="utf-8") as f:
        print("selected_words.json을 로드합니다.")
        selected_words = json.load(f)
else:
    print("selected_words.json을 생성합니다.")
    selected_words = [f[0] for f in text.vocab().most_common(FREQUENCY_COUNT)]
    with open('selected_words.json', 'w', encoding="utf-8") as make_file:
        json.dump(selected_words, make_file, ensure_ascii=False, indent="\t")

# 단어리스트 문서에서 상위 10000개들중 포함되는 단어들이 개수
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

import numpy as np
from keras.models import load_model

if os.path.isfile('results_1.h5'):
    print("results_1.h5를 로드합니다.")
    model = load_model('results_1.h5')

def predict_review(review):
    token = tokenize(review)
    tfq = term_frequency(token)
    data = np.expand_dims(np.asarray(tfq).astype('float32'), axis=0)
    score = float(model.predict(data))
    if (score > 0.5):
        print(f"{review} ==> 긍정 ({round(score * 100)}%)")
        emotion = 12
        return emotion
    else:
        print(f"{review} ==> 부정 ({round((1 - score) * 100)}%)")
        emotion = 34
        return emotion

import socket
from _thread import *


# 쓰레드에서 실행되는 코드입니다.

# 접속한 클라이언트마다 새로운 쓰레드가 생성되어 통신을 하게 됩니다.
def threaded(client_socket, addr):
    print('Connected by :', addr[0], ':', addr[1])

    # 클라이언트가 접속을 끊을 때 까지 반복합니다.
    while True:
        try:

            # 데이터가 수신되면 클라이언트에 다시 전송합니다.(에코)
            data = client_socket.recv(1024)

            if not data:
                print('Disconnected by ' + addr[0], ':', addr[1])
                break

            print('Received from ' + addr[0], ':', addr[1], data.decode())
            emotion = predict_review(data.decode())
            client_socket.send(emotion.to_bytes(4, byteorder='little'))

        except ConnectionResetError as e:
            print('Disconnected by ' + addr[0], ':', addr[1])
            break
    client_socket.close()

HOST = ''
PORT = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

print('server start')

# 클라이언트가 접속하면 accept 함수에서 새로운 소켓을 리턴합니다.

# 새로운 쓰레드에서 해당 소켓을 사용하여 통신을 하게 됩니다.
while True:
    print('wait')
    client_socket, addr = server_socket.accept()
    start_new_thread(threaded, (client_socket, addr))

server_socket.close()
