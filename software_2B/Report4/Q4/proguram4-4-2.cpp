#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <Eigen/Dense>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "使い方: " << argv[0] << " <点数N>=2以上 <分散sigma^2>=0以上\n";
        return 1;
    }
    
    const int N = std::atoi(argv[1]);
    const double sigma2 = std::atof(argv[2]);
    const double sigma = std::sqrt(sigma2);
    
    if (N < 2 || sigma2 < 0.0) {
        std::cerr << "エラー: 点数Nは2以上、分散sigma^2は0以上である必要があります。\n";
        return 1;
    }

    std::vector<double> x_noisy, y_noisy;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, sigma);

    std::cout << "=== 課題4: 最小二乗法による直線パラメータ推定 ===" << std::endl;
    std::cout << "データ点数: " << N << ", ノイズ分散: " << sigma2 << std::endl;
    std::cout << std::endl;

    // ノイズ付きデータ生成
    for (int alpha = 1; alpha <= N; ++alpha) {
        double x = -10.0 + (20.0 / (N - 1)) * (alpha - 1);
        double y = (3.0 * x + 4.0) / 2.0;
        double x_n = x + dist(gen);
        double y_n = y + dist(gen);
        x_noisy.push_back(x_n);
        y_noisy.push_back(y_n);
    }

    // 最小二乗法の準備
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_x  += x_noisy[i];
        sum_y  += y_noisy[i];
        sum_xx += x_noisy[i] * x_noisy[i];
        sum_xy += x_noisy[i] * y_noisy[i];
    }

    std::cout << "統計量:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Σx = " << sum_x << ", Σy = " << sum_y << std::endl;
    std::cout << "  Σx² = " << sum_xx << ", Σxy = " << sum_xy << std::endl;
    std::cout << std::endl;

    // 正規方程式をEigenで解く
    std::cout << "Eigenライブラリによる正規方程式の解法:" << std::endl;
    Eigen::Matrix2d A;
    A << sum_xx, sum_x,
         sum_x,  N;
    
    Eigen::Vector2d c;
    c << sum_xy,
         sum_y;

    std::cout << "行列A:" << std::endl << A << std::endl;
    std::cout << "ベクトルc:" << std::endl << c << std::endl;
    
    // 逆行列を使って解く
    Eigen::Vector2d ab = A.inverse() * c;
    double a = ab(0);
    double b = ab(1);

    // 結果表示
    std::cout << std::endl;
    std::cout << "=== 推定結果 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "a (傾き) = " << a << std::endl;
    std::cout << "b (切片) = " << b << std::endl;
    std::cout << "推定直線: y = " << a << "x + " << b << std::endl;
    
    // 理論値との比較
    std::cout << std::endl;
    std::cout << "=== 動作確認 ===" << std::endl;
    std::cout << "理論値: y = 1.5x + 2.0" << std::endl;
    std::cout << "推定値: y = " << a << "x + " << b << std::endl;
    std::cout << "傾き誤差: " << std::abs(a - 1.5) << std::endl;
    std::cout << "切片誤差: " << std::abs(b - 2.0) << std::endl;

    // ノイズ付きデータを出力
    std::ofstream ofs("data.tsv");
    if (!ofs) {
        std::cerr << "data.tsv の書き込みに失敗しました。\n";
        return 1;
    }
    ofs << "# x_noisy\ty_noisy\n";
    ofs << std::fixed << std::setprecision(5);
    for (int i = 0; i < N; ++i) {
        ofs << x_noisy[i] << "\t" << y_noisy[i] << "\n";
    }

    // 係数を Gnuplot で読み込める形式で保存
    std::ofstream pfs("params.dat");
    if (!pfs) {
        std::cerr << "params.dat の書き込みに失敗しました。\n";
        return 1;
    }
    pfs << "a = " << a << "\n";
    pfs << "b = " << b << "\n";
    
    std::cout << std::endl;
    std::cout << "data.tsv と params.dat を出力しました。\n";
    
    return 0;
}