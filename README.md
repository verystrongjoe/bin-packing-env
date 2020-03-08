# bin-packing-env
A reinforcement learning environment for bin packing with simple GUI. very flexible environment based on parameters


## 기존 2D Bin Packing Problem 단점
 - 소스나 구현체를 전혀 구할 수 없음
 - 논문마다 가정한 상황이 다 다름
 - 무엇보다 너무 간단. 
 - 물류 사업부의 복잡한 상황과는 난이도 괴리 심함

## 제약
여러분들이 생각하는 2D Bin Packing 문제에 기존의 물류 문제와 같은 제약 추가
- Bin아래 맞닿는 Bin들의 총 지지 단면적 설정(LOAD_WIDTH_THRESHOLD)
- 아래 BIN들의 하중의 제약이 있는 점을 착안하여(BIN마다 WEIGHT속성을 주어 새로 쌓이는 BIN들이 WEIGHT가 더 크면 못 쌓게함)
- 2D가 아닌 3D 상황을 감안하여 물건이 쌓을때 상단에 어떤 물체가 있으면 쌓지 못하게 하였음 (BIN 선택시 빈 공간 파악 필요)
- 당연히 BIN의 쌓는 위치에서의 크기가 팔레트 면적을 벗어나면 당연히 쌓지 못함

## 리워드
2가지 리워드를 주었음
ㅁ 리워드 모드 = 0  
 - 리워드는 놓았던 BIN의 총 개수 / 총 BIN의 개수
ㅁ 리워드 모드 = 1
-  총 BIN의 면적 

## 환경 파라메터

```
RENDER = True 팔레트 물건 선적 GUI를 생략 ( 더 빠름)
TICK_INTERVAL = 20  # the smaller it is, the slower the game plays
CELL_SIZE = 20 # 팔레트 단위면적(셀) 크기

PALLET SIZE 
ROW_COUNT = 30
COL_COUNT = 30

# Bins information
BIN_MAX_COUNT = 50
EPISODE_MAX_STEP = 50

BIN_MIN_X_SIZE = 1
BIN_MIN_Y_SIZE = 1
BIN_MAX_X_SIZE = 10
BIN_MAX_Y_SIZE = 10
BIN_MIN_W_SIZE = 1
BIN_MAX_W_SIZE = 1
LOAD_WIDTH_THRESHOLD = 0.8  # Ratio
```


##  에이전트 
- Random Agent / DQN Agent 
작동하게 만들어 놨으며 DQN으로는 학습이 간단한건 될 것 같으나 진행 예정.
ㅁState 
 - Bins' List , 2D Pallet 의 Bin 인덱스 + 하중 
ㅁReward
 - 리워드

## Action
크게 3가지로 나뉘어짐 ( Bin의 Index, X/Y축 탐색 우선순위, Rotation)
 - Action은 x,y를 선택하게 할려고 했으나, 팔레트 사이즈마다 너무 다르게 정의가 되고 또 경우의 수가 많은 점을 감안하고 문제자체도 너무 어려운 점을 감안하여, x축 기준으로 가장 가능한점을 먼저 찾을 것이냐, 아니면 y축을 먼저 기준으로 찾을 거냐 선택을 하도록 정의하였으며, 마지막으로 width와 height를 바꿀수 있는 rotate를 정의 가능하다

##  업그레이드 계획
- 스레드별 환경 로드(Learning with Multiple Actors 지원)
- setup.py 및 pip install 하게끔 업로드
- 물류 문제처럼 팔레트가 여러개 있다고 가정 
- N개 이상 팔레트가 있다하고, 그만큼의 BIN들이 주고나서 N개의 팔렛트를 채울때까지 하나의 에피소드로 묶음
- 다른 알고리즘과 비교 검증시를 위해서 Random으로 BIn을 생성도 하지만 별도의 csv로 읽어서 그 기준으로 생성하게 변경
- DQN Agent 알고리즘 만들고 간단한 환경에서 학습 가능한지 돌려볼 계획
- pytorch TensorboardX를 만들어서 간단한 실험결과를 넣어 튜토리얼 작성
