import os
print("현재 작업 디렉토리:", os.getcwd())

file_path = './example/mESC/refNetwork.csv'

# 경로와 파일이 존재하는지 확인
if os.path.exists(file_path):
    print("파일이 존재합니다.")
else:
    print("파일을 찾을 수 없습니다. 경로를 확인해 주세요.")