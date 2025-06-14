#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <string.h>

int main(int argc, char* argv[]) {
    // 1. 引数チェック（課題要求: 引数でディレクトリ名を受け取る）
    if (argc != 2) {
        // 引数の数が正しくない場合のエラーメッセージ
        std::cerr << "エラー: 引数の数が正しくありません。" << std::endl;
        std::cerr << "使用法: " << argv[0] << " <ディレクトリ名>" << std::endl;
        return 1;
    }

    // 2. vectorの準備（課題要求: vectorに格納）
    std::vector<std::string> filenames;

    // 3. ディレクトリ走査
    DIR* dir = opendir(argv[1]);  // 指定したディレクトリをオープンする
    struct dirent* file;
    while ((file = readdir(dir)) != nullptr) {  // opendirで取得したDIR構造体からファイル名を順番に取得する
        if (strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0) {
            filenames.push_back(file->d_name);  // vectorに格納

            std::cout << file->d_name << std::endl;  // 即座に出力（授業資料参考）
        }
    }
    closedir(dir);  // オープンしたディレクトリをクローズする

    // 4. 動作確認
    std::cout << "=== 課題1 動作確認 ===" << std::endl;
    std::cout << "ディレクトリ: " << argv[1] << std::endl;
    std::cout << "ファイル数: " << filenames.size() << std::endl;
    std::cout << "ファイル一覧:" << std::endl;
    
    for (size_t i = 0; i < filenames.size(); i++) {
        std::cout << "  [" << i << "] " << filenames[i] << std::endl;
    }

    return 0;
}