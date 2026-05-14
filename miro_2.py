import numpy    # 미로 맵 제작
import random   # 랜덤이동
import time     # 딜레이

# numpy 행렬 기준 : (0,0) → col(y) + n / (0,0) ↓ row(x) + n
miro = numpy.array([
    [1,2,1,1,1,1,1,1,1,1], # ■▽■■■■■■■■
    [1,0,0,0,0,0,0,0,0,1], # ■□□□□□□□□■
    [1,0,1,1,0,1,1,1,1,1], # ■□■■□■■■■■
    [1,0,1,1,0,0,0,0,0,1], # ■□■■□□□□□■
    [1,0,1,1,1,1,1,1,0,1], # ■□■■■■■■□■
    [1,0,0,0,0,0,0,1,0,1], # ■□□□□□□■□■
    [1,0,1,1,1,0,1,1,0,1], # ■□■■■□■■□■
    [1,0,0,0,1,0,0,1,1,1], # ■□□□■□□■■■
    [1,0,1,0,1,1,0,0,0,1], # ■□■□■■□□□■
    [1,1,1,1,1,1,1,1,3,1], # ■■■■■■■■▼■
])

fail_miro = numpy.array([
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1],  # ■■■■■■■■■■
    [1,1,1,1,1,1,1,1,1,1]   # ■■■■■■■■■■
])

end_miro = numpy.array([
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
    [0,0,0,0,0,0,0,0,0,0],  # □□□□□□□□□□
])

# 미로 딕셔너리
# 0, □ : 길 / 1, ■ : 벽 / 2, ▽ : 시작 / 3, ▼ : 도착 / 4, ☆ : 현재 위치
mr = {0: '□', 1: '■', 2: '▽', 3: '▼', 4: '☆'}

# 시작 위치 찾기
start = numpy.where(miro == 2)
start_x, start_y = start[0][0], start[1][0]

# Q 테이블(+무한대)로 초기화
Q_table = numpy.zeros((10, 10, 4))  # (10,10,4) == (가로길이, 세로길이, wasd)

# wsad : 상하좌우
wsad = [(-1,0), (1,0), (0,-1), (0,1)]

# 학습 파라미터, 1==100%
alpha = 0.15        # 학습률(값이 클수록 빠르고 불안정하게 학습)
gamma = 0.85        # 감가율(미래 보상 중요도)
epsilon = (0.999)   # 탐험률(새로운 길 선택 확률 ex. 돌연변이)
episodes = 1000     # 에피소드(시도 및 학습을 n회 반복)
'''
# 현재 위치(h_x, h_y)에 ☆ 표시
def print_miro(miro, h_x=None, h_y=None) :
    for i, row in enumerate(miro) :
        line = []
        for ii, cell in enumerate(row) :
            if (h_x is not None and h_y is not None and i == h_x and ii == h_y) :
                line.append(mr[4])  # 현재 위치에 ☆ 출력
            else:
                line.append(mr[cell])
        print(" ".join(line))
    print()
'''
# 벽이 아닌 길 찾기
def can_r(x, y):
    return 0 <= x < 10 and 0 <= y < 10 and miro[x][y] != 1

success_episodes = []       # 성공한 에피소드 리스트

# Q러닝 학습
for episode in range(episodes):
    x, y = start_x, start_y  # 매번 시작 위치로 초기화
    steps = 0   # 몇 번 이동했는지

    for _ in range(100):  # 100회 이동 시 코드 중단

        #print_miro(fail_miro)
        #time.sleep(0.2)
        
        can_actions = []
        
        for idx, (dx, dy) in enumerate(wsad):
            nx_temp, ny_temp = x + dx, y + dy
            if can_r(nx_temp, ny_temp):
                can_actions.append(idx)
        if not can_actions:
            continue
        if random.random() < epsilon:   # 0 ~ 1 사이 랜덤 숫자 중 0.2보다 작으면 탐험험
            wsad_idx = random.choice(can_actions) # 0 ~ 3 사이 랜덤  숫자를 고름 (상하좌우)
        else:
            q_values = [Q_table[x][y][i] if i in can_actions else -numpy.inf for i in range(4)]   # (Q값 = -무한대)벽 방향 선택 x
            wsad_idx = numpy.argmax(q_values)   # 현재 가장 높은 Q값을 가진 행동 시행

        # 이동
        dx, dy = wsad[wsad_idx]
        nx, ny = x + dx, y + dy

        # 보상 계산
        if miro[nx][ny] == 3:
            reward = 1000  # 도착 시 +1000
        else:
            reward = -10   # 한 칸 이동 시 -10

        # Q 업데이트
        # Q(현재상태,선택한행동)=Q(현재상태,선택한행동)+학습률×[보상+감가율×다음상태에서의최고Q값−기존Q값]
        Q_table[x][y][wsad_idx] += alpha * (
            reward + gamma * numpy.max(Q_table[nx][ny]) - Q_table[x][y][wsad_idx]
        )

        x, y = nx, ny
        steps += 1

        # 현재 위치에 ☆ 표시
        #print_miro(miro, x, y)
        #time.sleep(0.2)

        if miro[x][y] == 3:  # 도착한다면
            success_episodes.append((episode + 1, steps))  # (회차, 이동 횟수) 저장
            #print_miro(end_miro) 
            print(f"[{episode+1}] 회차 성공")
            print(f"현재 이동 횟수: {steps}회")
            break

    epsilon = max(0.05, epsilon * 0.9995) # 탐험률 점진적 감소 (최소 0.05)

x, y = start_x, start_y # 시작점 좌표로 초기화
for i in range(100):    # 초반 무한루프 방지
    #print_miro(miro, x, y)
    #time.sleep(0.2)

    wsad_idx = numpy.argmax(Q_table[x][y])  # 높은 Q값 선택
    dx, dy = wsad[wsad_idx] # 방향 벡터
    nx, ny = x + dx, y + dy # 위치 계산

    x, y = nx, ny   # 현재 좌표 = 다음 좌표

    if miro[x][y] == 3: # 도착한다면
        #print_miro(end_miro)
        #print_miro(miro, x, y)
        break   # 종료 후 다음 회차로

#print_miro(miro)

if success_episodes:
    print("성공한 에피소드 목록:")
    for episode_num, step_count in success_episodes:
        print(f"{episode_num}회차: {step_count}번 이동")

    best_episode = min(success_episodes, key=lambda x: x[1])    # 가장 적게 이동한 에피소드를 success_episodes에서 불러옴
    print(f"가장 적게 이동한 회차: {best_episode[0]}회차 {best_episode[1]}번 이동")
else:
    print("도착한 회차(에피소드) 없음")