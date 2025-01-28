def create_small_file(input_file, output_file, line_count=128000):
    try:
        # 입력 파일 읽기
        with open(input_file, 'r') as infile:
            # 출력 파일 쓰기
            with open(output_file, 'w') as outfile:
                for i in range(line_count):
                    line = infile.readline()
                    if not line:
                        break  # 파일의 끝에 도달하면 종료
                    outfile.write(line)
        print(f"{output_file} 생성 완료.")
    except Exception as e:
        print(f"오류 발생: {e}")

# 사용 예시
input_filename = 'tsp100_train_concorde.txt'  # 원본 파일명
output_filename = 'tsp100_train_concorde_small.txt'  # 생성할 작은 파일명

create_small_file(input_filename, output_filename)
