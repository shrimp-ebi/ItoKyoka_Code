#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    // パラメータ設定
    const int N = 100;
    const double sigma_squared = 0.25;
    const double sigma = 0.5;  // sqrt(0.25)
    const int seed = 42;

    // 乱数生成器
    std::mt19937 generator(seed);
    std::normal_distribution<double> noise(0.0, sigma);

    // データを格納するベクトル
    std::vector<double> x_ideal(N), y_ideal(N);
    std::vector<double> x_noisy(N), y_noisy(N);

    std::cout << "直線上の点生成と誤差付加" << std::endl;
    std::cout << "N = " << N << ", σ² = " << sigma_squared << std::endl;
    std::cout << std::endl;

    // 1. 式(1)による理想的な点の生成
    for (int alpha = 1; alpha <= N; alpha++) {
        int i = alpha - 1;
        x_ideal[i] = -10.0 + (20.0 / (N - 1)) * (alpha - 1);
        y_ideal[i] = (3.0 * x_ideal[i] + 4.0) / 2.0;
    }

    // 2. 誤差の付加
    for (int i = 0; i < N; i++) {
        x_noisy[i] = x_ideal[i] + noise(generator);
        y_noisy[i] = y_ideal[i] + noise(generator);
    }

    // 結果表示（最初の10点）
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "最初の10点の結果:" << std::endl;
    std::cout << "α    理想座標           誤差付き座標" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << (i+1) << "   (" << std::setw(6) << x_ideal[i] << ", " 
                  << std::setw(6) << y_ideal[i] << ")   (" 
                  << std::setw(6) << x_noisy[i] << ", " 
                  << std::setw(6) << y_noisy[i] << ")" << std::endl;
    }

    // 直線方程式の確認
    std::cout << std::endl << "直線方程式の確認 (3x - 2y + 4 = 0):" << std::endl;
    for (int i = 0; i < 5; i++) {
        double check = 3.0 * x_ideal[i] - 2.0 * y_ideal[i] + 4.0;
        std::cout << "点" << (i+1) << ": " << check << std::endl;
    }

    return 0;
}