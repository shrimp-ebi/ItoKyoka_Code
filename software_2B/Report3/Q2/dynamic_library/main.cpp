#include <iostream>
#include "Message.h"

int main() {
    Message msg;
    msg.addMessage("動的ライブラリのテスト");
    msg.addMessage("Hello Dynamic Library");
    
    std::cout << "=== 動的ライブラリ動作確認 ===" << std::endl;
    msg.showAllMessages();
    std::cout << "メッセージ数: " << msg.getNMessages() << std::endl;
    
    return 0;
}