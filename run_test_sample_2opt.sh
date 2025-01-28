#!/bin/bash

# 실행할 스크립트 리스트

for SCRIPT in \
    "./runs_test_sample_2opt/test_tsp50_blackout.sh" \
    "./runs_test_sample_2opt/test_tsp50_categorical.sh" \
    "./runs_test_sample_2opt/test_tsp50_improved_blackout.sh" \
    "./runs_test_sample_2opt/test_tsp50_more_improved_blackout.sh" \
    "./runs_test_sample_2opt/test_tsp100_blackout.sh" \
    "./runs_test_sample_2opt/test_tsp100_categorical.sh" \
    "./runs_test_sample_2opt/test_tsp100_improved_blackout.sh" \
    "./runs_test_sample_2opt/test_tsp100_more_improved_blackout.sh" \
    "./runs_test_sample_2opt/test_tsp500_blackout.sh" \
    "./runs_test_sample_2opt/test_tsp500_categorical.sh" \
    "./runs_test_sample_2opt/test_tsp500_improved_blackout.sh" \
    "./runs_test_sample_2opt/test_tsp500_more_improved_blackout.sh"

# 스크립트 실행 및 개별 로그 파일 생성
do
    if [ -f "$SCRIPT" ]; then
        LOG_FILE="${SCRIPT%.*}_log.txt"
        echo "Running $SCRIPT..." | tee -a "$LOG_FILE"
        
        START_TIME=$(date +%s)

        # 스크립트 실행, 출력 및 에러를 로그 파일에 저장
        sh "$SCRIPT" >> "$LOG_FILE" 2>&1

        END_TIME=$(date +%s)
        EXECUTION_TIME=$((END_TIME - START_TIME))

        echo "Execution time for $SCRIPT: ${EXECUTION_TIME} seconds" | tee -a "$LOG_FILE"
        echo "$SCRIPT completed in ${EXECUTION_TIME} seconds." | tee -a "$LOG_FILE"
    else
        echo "Script $SCRIPT not found. Skipping..."
    fi
done
