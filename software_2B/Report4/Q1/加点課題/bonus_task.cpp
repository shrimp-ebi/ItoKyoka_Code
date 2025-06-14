#include <iostream>
#include <random>
#include <vector>
#include <fstream>
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
    std::mt19937 generator(seed);
    std::normal_distribution<double> dist(mu, sigma);
    
    std::vector<double> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = dist(generator);
    }
    return data;
}

// 単一のNに対する統計量を計算
struct Statistics {
    double mean;
    double variance;
    double mean_error;    // 理論平均からの誤差
    double variance_error; // 理論分散からの誤差
};

Statistics calculate_statistics_for_N(int N, double mu, double sigma, int seed) {
    // データ生成
    std::vector<double> data = generate_normal_data(mu, sigma, N, seed);
    
    // 統計量計算
    double sample_mean = compute_sample_mean(data);
    double sample_variance = compute_sample_variance(data, sample_mean);
    
    // 理論値からの誤差
    double theoretical_variance = sigma * sigma;
    double mean_error = std::abs(sample_mean - mu);
    double variance_error = std::abs(sample_variance - theoretical_variance);
    
    return {sample_mean, sample_variance, mean_error, variance_error};
}

int main() {
    // パラメータ設定
    const double mu = 5.0;           // 固定平均
    const double sigma = 2.0;        // 固定標準偏差
    const int base_seed = 42;        // 基準シード値
    
    // 試行するデータ数のリスト
    std::vector<int> N_values = {10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
    
    // 結果をファイルに出力
    std::ofstream output_file("statistics_vs_N.dat");
    if (!output_file) {
        std::cerr << "エラー: ファイルを開けませんでした" << std::endl;
        return 1;
    }
    
    // ヘッダー行
    output_file << "# N\tsample_mean\tsample_variance\tmean_error\tvariance_error" << std::endl;
    output_file << std::fixed << std::setprecision(6);
    
    std::cout << "=== データ数Nによる統計量の変化分析 ===" << std::endl;
    std::cout << std::endl;
    std::cout << "固定パラメータ:" << std::endl;
    std::cout << "  理論平均 μ = " << mu << std::endl;
    std::cout << "  理論標準偏差 σ = " << sigma << std::endl;
    std::cout << "  理論分散 σ² = " << sigma * sigma << std::endl;
    std::cout << std::endl;
    
    std::cout << std::setw(8) << "N" 
              << std::setw(15) << "標本平均"
              << std::setw(15) << "標本分散"
              << std::setw(15) << "平均誤差"
              << std::setw(15) << "分散誤差" << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    
    // 各Nに対して計算
    for (int N : N_values) {
        Statistics stats = calculate_statistics_for_N(N, mu, sigma, base_seed + N);
        
        // ファイルに出力（タブ区切り）
        output_file << N << "\t" 
                   << stats.mean << "\t" 
                   << stats.variance << "\t"
                   << stats.mean_error << "\t"
                   << stats.variance_error << std::endl;
        
        // 画面に出力
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(8) << N
                  << std::setw(15) << stats.mean
                  << std::setw(15) << stats.variance
                  << std::setw(15) << stats.mean_error
                  << std::setw(15) << stats.variance_error << std::endl;
    }
    
    output_file.close();
    
    std::cout << std::endl;
    std::cout << "結果ファイル 'statistics_vs_N.dat' を生成しました。" << std::endl;
    std::cout << std::endl;
    std::cout << "gnuplotでグラフを作成するには:" << std::endl;
    std::cout << "  gnuplot plot_script.gp" << std::endl;
    
    return 0;
}