#!/bin/bash

# Fuji.jpgに対する系統的実験
SCRIPT="gauss.py"
IMAGE="Fuji.jpg"
SCALE_TRUE=1.2
THETA_TRUE=5
THRESHOLD=1e-5
MAX_LOOP=1000

echo "=== Fuji.jpgでの初期値依存性実験 ==="

# 1. スケール初期値の影響（角度は真値付近）
echo "--- スケール初期値実験 ---"
for SCALE_INIT in 0.5 0.8 1.0 1.2 1.5 2.0; do
    echo "実行: scale_init=$SCALE_INIT, theta_init=5"
    python $SCRIPT $IMAGE $SCALE_TRUE $THETA_TRUE --scale_init $SCALE_INIT --theta_init 5 --threshold $THRESHOLD --max_loop $MAX_LOOP --output_path output/fuji_scale_init
done

# 2. 角度初期値の影響（スケールは真値付近）
echo "--- 角度初期値実験 ---"
for THETA_INIT in -20 -10 0 5 15 30 45; do
    echo "実行: scale_init=1.2, theta_init=$THETA_INIT"
    python $SCRIPT $IMAGE $SCALE_TRUE $THETA_TRUE --scale_init 1.2 --theta_init $THETA_INIT --threshold $THRESHOLD --max_loop $MAX_LOOP --output_path output/fuji_theta_init
done

# 3. 両方を変化させた実験（より詳細）
echo "--- 組み合わせ実験 ---"
for SCALE_INIT in 0.8 1.0 1.2 1.5; do
    for THETA_INIT in 0 5 15 30; do
        echo "実行: scale_init=$SCALE_INIT, theta_init=$THETA_INIT"
        python $SCRIPT $IMAGE $SCALE_TRUE $THETA_TRUE --scale_init $SCALE_INIT --theta_init $THETA_INIT --threshold $THRESHOLD --max_loop $MAX_LOOP --output_path output/fuji_combined
    done
done

# 4. 極端な初期値での実験
echo "--- 極端な初期値実験 ---"
for SCALE_INIT in 0.1 0.3 3.0 5.0; do
    for THETA_INIT in -90 -45 60 90; do
        echo "実行: scale_init=$SCALE_INIT, theta_init=$THETA_INIT"
        python $SCRIPT $IMAGE $SCALE_TRUE $THETA_TRUE --scale_init $SCALE_INIT --theta_init $THETA_INIT --threshold $THRESHOLD --max_loop $MAX_LOOP --output_path output/fuji_extreme
    done
done

echo "実験完了"