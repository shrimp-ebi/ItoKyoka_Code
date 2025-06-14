#include <iostream>
#include <Eigen/Dense>

int main() {
    // ベクトルの定義
    Eigen::VectorXd a(3);
    Eigen::VectorXd b(3);
    a << 1, 2, 3;
    b << 4, 5, 6;
    
    // 行列の定義
    Eigen::MatrixXd A(3, 3);
    Eigen::MatrixXd B(3, 3);
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    B << 9, 8, 7,
         6, 5, 4,
         3, 2, 1;



    // 入力値の表示
    std::cout << "入力データ:" << std::endl;
    std::cout << "ベクトル a:" << std::endl;
    std::cout << a << std::endl;
    std::cout << std::endl;
    
    std::cout << "ベクトル b:" << std::endl;
    std::cout << b << std::endl;
    std::cout << std::endl;
    
    std::cout << "行列 A:" << std::endl;
    std::cout << A << std::endl;
    std::cout << std::endl;
    
    std::cout << "行列 B:" << std::endl;
    std::cout << B << std::endl;
    std::cout << std::endl;



    // 演算の実行と結果の表示
    std::cout << "=== 演算結果 ===" << std::endl;
    // 1. ベクトル同士の和 a + b
    std::cout << "ベクトル同士の和 a + b:" << std::endl;
    std::cout << a + b << std::endl;
    std::cout << std::endl;

    // 2. ベクトル同士の積 ab⊤
    std::cout << "ベクトル同士の積 ab⊤:" << std::endl;
    std::cout << a * b.transpose() << std::endl;
    std::cout << std::endl;

    // 3. ベクトルの内積 a⊤b
    std::cout << "ベクトルの内積 a⊤b:" << std::endl;
    std::cout << a.dot(b) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    // 4. 行列とベクトルの積 Ab
    std::cout << "行列とベクトルの積 Ab:" << std::endl;
    std::cout << A * b << std::endl;
    std::cout << std::endl;

    // 5. 行列同士の積 AB
    std::cout << "行列同士の積 AB:" << std::endl;
    std::cout << A * B << std::endl;
    std::cout << std::endl;

    return 0;
}