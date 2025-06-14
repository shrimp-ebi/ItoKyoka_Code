#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "課題4: 最小二乗法による直線パラメータ推定" << std::endl;
    std::cout << std::endl;

    // 課題3のデータを読み込み
    std::ifstream file("task3_data.dat");
    if (!file.is_open()) {
        std::cerr << "エラー: task3_data.dat が見つかりません" << std::endl;
        return 1;
    }

    std::vector<double> x_noisy, y_noisy;
    std::string line;
    std::getline(file, line); // ヘッダーをスキップ
    
    // ノイズ付きの点 (xn, yn) のみを使って最小二乗法を適用
    double xi, yi, xn, yn;
    while (file >> xi >> yi >> xn >> yn) {
        x_noisy.push_back(xn);
        y_noisy.push_back(yn);
    }
    file.close();

    int N = x_noisy.size();
    std::cout << "データ点数: " << N << std::endl;
    std::cout << std::endl;

    
    // 必要な統計量を計算
    std::cout << "ステップ1: 正規方程式に必要な統計量を計算" << std::endl;
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    
    for (int alpha = 0; alpha < N; alpha++) {
        sum_x += x_noisy[alpha];
        sum_y += y_noisy[alpha];
        sum_xx += x_noisy[alpha] * x_noisy[alpha];
        sum_xy += x_noisy[alpha] * y_noisy[alpha];
    }

    std::cout << "計算した統計量:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Σxα    = " << std::setw(12) << sum_x << std::endl;
    std::cout << "  Σyα    = " << std::setw(12) << sum_y << std::endl;
    std::cout << "  Σxα²   = " << std::setw(12) << sum_xx << std::endl;
    std::cout << "  Σxαyα  = " << std::setw(12) << sum_xy << std::endl;
    std::cout << "  N      = " << std::setw(12) << N << std::endl;
    std::cout << std::endl;

    // 正規方程式の表示
    std::cout << "ステップ2: 正規方程式を構成" << std::endl;
    std::cout << "  (Σxα²  Σxα ) (a)   (Σxαyα)" << std::endl;
    std::cout << "  (Σxα   N   ) (b) = (Σyα  )" << std::endl;
    std::cout << std::endl;
    std::cout << "  数値を代入すると:" << std::endl;
    std::cout << "  (" << std::setw(8) << sum_xx << " " << std::setw(8) << sum_x << ") (a)   (" << std::setw(8) << sum_xy << ")" << std::endl;
    std::cout << "  (" << std::setw(8) << sum_x << " " << std::setw(8) << N << ") (b) = (" << std::setw(8) << sum_y << ")" << std::endl;
    std::cout << std::endl;

    // 連立方程式を解く
    std::cout << "ステップ3: クラメルの公式で連立方程式を解く" << std::endl;
    double det = sum_xx * N - sum_x * sum_x;
    std::cout << "  行列式 det = " << sum_xx << "×" << N << " - " << sum_x << "×" << sum_x << " = " << det << std::endl;
    
    double a = (N * sum_xy - sum_x * sum_y) / det;
    double b = (sum_xx * sum_y - sum_x * sum_xy) / det;
    
    std::cout << "  a = (" << N << "×" << sum_xy << " - " << sum_x << "×" << sum_y << ") / " << det << " = " << a << std::endl;
    std::cout << "  b = (" << sum_xx << "×" << sum_y << " - " << sum_x << "×" << sum_xy << ") / " << det << " = " << b << std::endl;
    std::cout << std::endl;

    std::cout << "===========================================" << std::endl;
    std::cout << "■ 最小二乗法の結果" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << std::setprecision(8);
    std::cout << "推定された直線の方程式: y = ax + b" << std::endl;
    std::cout << std::endl;
    std::cout << "  a (傾き) = " << std::setw(12) << a << std::endl;
    std::cout << "  b (切片) = " << std::setw(12) << b << std::endl;
    std::cout << std::endl;
    std::cout << "推定直線: y = " << a << "x + " << b << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << std::endl;

   // 理論値との比較
    std::cout << "■ 動作確認: 理論値との比較" << std::endl;
    double a_theory = 1.5;  // 元の直線 3x - 2y + 4 = 0 から y = 1.5x + 2
    double b_theory = 2.0;
    
    std::cout << "  理論値（元の直線）: y = " << a_theory << "x + " << b_theory << std::endl;
    std::cout << "  推定値（最小二乗）: y = " << a << "x + " << b << std::endl;
    std::cout << std::endl;


    // 目的関数 J の値を計算
    std::cout << "■ 最適化の確認" << std::endl;
    double J = 0.0;
    for (int alpha = 0; alpha < N; alpha++) {
        double error = y_noisy[alpha] - a * x_noisy[alpha] - b;
        J += 0.5 * error * error;
    }
    std::cout << "  目的関数の値 J = (1/2)Σ(yα - axα - b)² = " << J << std::endl;
    std::cout << std::endl;

    return 0;
}