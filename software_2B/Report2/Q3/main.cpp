#include <iostream>
#include "Message.h"

int main() {
    std::cout << "=== 課題3 動作確認 ===" << std::endl;
    
    // 1. デフォルトコンストラクタのテスト
    Message msg1; // 空のMessageオブジェクトを作成

    // メッセージを追加
    msg1.addMessage("Hello");  // 機能1
    msg1.addMessage("World");
    msg1.addMessage("C++");

    // 出力
    std::cout << "msg1のメッセージ数: " << msg1.getNMessages() << std::endl;  // 機能4
    std::cout << "msg1の全メッセージ:" << std::endl;
    msg1.showAllMessages();  // 機能3
    std::cout << "msg1の1番目: " << msg1.getMessage(1) << std::endl; // 機能2
    


    // 2. 文字列で初期化するコンストラクタのテスト
    Message msg2("初期メッセージ");  // 最初から1つのメッセージを持つMessageオブジェクトを作成
    msg2.addMessage("追加メッセージ");  // メッセージを追加
    
    // 出力
    std::cout << "\nmsg2の内容:" << std::endl;
    msg2.showAllMessages();
    


    // 3. vectorで初期化するコンストラクタのテスト
    std::vector<std::string> initial = {"メッセージ1", "メッセージ2", "メッセージ3"};  // 初期メッセージのvectorを作成
    Message msg3(initial);  // 最初から3つのメッセージを持つMessageオブジェクトを作成

    // 出力
    std::cout << "\nmsg3の内容:" << std::endl;
    msg3.showAllMessages();
    std::cout << "msg3のメッセージ数: " << msg3.getNMessages() << std::endl;
    
    return 0;
}