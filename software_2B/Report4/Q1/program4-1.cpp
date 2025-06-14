#include <iostream>
#include <random>
#include <vector>
#include <cstdlib>
#include <iomanip>

// 標本平均を計算する関数
double compute_sample_mean(const std::vector<double>& data) {
    double sum = 0.0;
    for (const auto& x : data) {
        sum += x;
    }
    return sum / data.size();
}

// 標本分散を計算する関数
double compute_sample_variance(const std::vector<double>& data, double mean) {
    double sum_squared_diff = 0.0;
    for (const auto& x : data) {
        double diff = x - mean;
        sum_squared_diff += diff * diff;
    }
    return sum_squared_diff / (data.size() - 1);  // 不偏推定量
}

// 乱数データを生成する関数
std::vector<double> generate_normal_data(double mu, double sigma, int N, int seed) {
    std::mt19937 generator(seed);                        // メルセンヌ・ツイスタ生成器
    std::normal_distribution<double> dist(mu, sigma);    // 正規分布
    
    std::vector<double> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = dist(generator);
    }
    return data;
}

// 入力値を検証する関数
bool validate_input(double sigma, int N) {
    if (sigma <= 0) {
        std::cerr << "エラー: 標準偏差は正の値である必要があります" << std::endl;
        return false;
    }
    if (N <= 1) {
        std::cerr << "エラー: データ数は2以上である必要があります" << std::endl;
        return false;
    }
    return true;
}

// 結果を出力する関数
void print_results(double mu, double sigma, int N, int seed, 
                   double sample_mean, double sample_variance) {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== 正規分布乱数生成と統計計算 ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "生成パラメータ:" << std::endl;
    std::cout << "  平均 μ = " << mu << std::endl;
    std::cout << "  標準偏差 σ = " << sigma << std::endl;
    std::cout << "  データ数 N = " << N << std::endl;
    std::cout << "  シード値 = " << seed << std::endl;
    std::cout << std::endl;
    
    std::cout << "計算結果:" << std::endl;
    std::cout << "  標本平均 = " << sample_mean << std::endl;
    std::cout << "  標本分散 = " << sample_variance << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // 引数チェック
    if (argc != 5) {
        std::cerr << "使用法: " << argv[0] << " μ σ N seed" << std::endl;
        std::cerr << "  μ: 正規分布の平均" << std::endl;
        std::cerr << "  σ: 正規分布の標準偏差" << std::endl;
        std::cerr << "  N: 生成するデータ数" << std::endl;
        std::cerr << "  seed: 乱数のシード値" << std::endl;
        return 1;
    }

    // コマンドライン引数の取得
    double mu = std::atof(argv[1]);      // 平均
    double sigma = std::atof(argv[2]);   // 標準偏差
    int N = std::atoi(argv[3]);          // データ数
    int seed = std::atoi(argv[4]);       // シード値

    // 入力値の検証
    if (!validate_input(sigma, N)) {
        return 1;
    }

    // データ生成
    std::vector<double> data = generate_normal_data(mu, sigma, N, seed);

    // 統計量の計算
    double sample_mean = compute_sample_mean(data);
    double sample_variance = compute_sample_variance(data, sample_mean);

    // 結果の出力
    print_results(mu, sigma, N, seed, sample_mean, sample_variance);

    return 0;
}