#include <iostream>
#include "Message.h"

// デフォルトコンストラクタ
Message::Message() {
    // vectorは自動で空の状態で初期化される
}

// 1つの文字列で初期化するコンストラクタ
Message::Message(const std::string& message_string) {
    message.push_back(message_string);
}

// vectorで初期化するコンストラクタ
Message::Message(const std::vector<std::string>& message_vector) {
    message = message_vector;
}

// デストラクタ
Message::~Message() {
    // vectorとstringは自動でメモリ管理されるため、特に何もしなくて良い
}



/* -------------- 指定機能-------------- */

// 文字列を登録
void Message::addMessage(const std::string& message_string) {
    message.push_back(message_string);
}

// 指定IDに対応する文字列を関数の戻り値として返す
std::string Message::getMessage(int message_id) {
    if (message_id >= 0 && message_id < message.size()) {
        return message[message_id];
    }
    return "";  // 範囲外なら空文字列
}

// 保持しているすべての文字列を表示する
void Message::showAllMessages(void) {
    for (int i = 0; i < message.size(); i++) {
        std::cout << i << ": " << message[i] << std::endl;
    }
}

// 保持している文字列数を関数の戻り値として返す
int Message::getNMessages(void) {
    return message.size();
}