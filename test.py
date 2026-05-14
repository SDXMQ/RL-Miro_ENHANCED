import numpy
import time     # 딜레이
import random   # 랜덤 이동
# 최단 이동 횟수 17 (시작, 도착 포함)

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

def print_miro(miro):
    mr = {0: '□', 1: '■', 2: '▽', 3: '▼', 4: '☆'}  # 미로 성분 딕셔너리
    # 0, □ : 길 / 1, ■ : 벽 / 2, ▽ : 시작 / 3, ▼ : 도착 / 4, ☆ : 현재 위치치
    for row in miro :    # 각 가로줄(row) 반복
        print(" ".join(mr[cell] for cell in row)) 
        # for cell in row 한 줄(row)의 각 칸(cell)을 가져옴
        # 각 칸을 ""로 구분하여 문자열로 변환
        # mr[cell] 각 숫자에 해당하는 기호로 변환

# 시작 위치(2,▽)를 찾는다(시작한다).
start_l = numpy.where(miro == 2)
# start_l는 튜플 형태로 (행, 열) 배열을 반환함. 첫 번째 시작 위치(2,▽)를 선택
x, y = start_l[0][0], start_l[1][0]  # x: 행, y: 

# 시작 위치 출력
print(f"시작 위치: ({x}, {y})") # f를 쓰면 ""안에 수식을 넣을 수 있다?!(두둥탁)

print_miro(miro)

# 랜덤 이동
def random_move() :
    global x, y     # 현재 위치 갱신

    # numpy 행렬 기준 : (0,0) → col(y) + n / (0,0) ↓ row(x) + n
    moves = {
        'w' : (-1,0),
        's' : (1,0),
        'a' : (0,-1),
        'd' : (0,1)
    }

    can_moves = [] # 현재 위치, 이동 가능한 방향 갱신신

    # 벽이 아닌 길 찾기
    for direction, (dx, dy) in moves.items() :
        can_x, can_y = x + dx, y + dy
        if ( 0 <= can_x < miro.shape[0] and 
            0 <= can_y < miro.shape[1] and 
            miro[can_x, can_y] != 1 and 
            miro[can_x, can_y] != 2 
            ) :
            can_moves.append((direction, can_x, can_y))  # 만약 벽이 아닐 경우 그 좌표를 추가

    # 이동 가능한 방향으로 이동
    if can_moves :
        direction, can_x, can_y = random.choice(can_moves)  # 랜덤 방향 선택
        
        if miro[x,y] != 2 : # 시작점 (2) 유지지
            miro[x,y] = 0   # 기존 위치를 0으로 변경
        
        x, y = can_x, can_y # 새로운 위치로 이동
        
        if miro[x,y] != 3 : # 도착지점 3
            miro[x,y] = 4   # 이동한 좌표에 4☆ 표시

for i in range(100) :   # 만약 100번 이상 이동하면 강제 종료
    random_move()
    print_miro(miro)
    time.sleep(0.1)

    if miro[x,y] == 3 : # 3▼에 도착 한다면
        miro[x,y] = 4  # 탈출 시각화
        print(i+1, "의 시도로 탈출")
        break

else :
    print("100번 초과")