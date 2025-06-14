#include <iostream>
#include "Message.h"

int main() {
    Message msg;
    msg.addMessage("静的ライブラリのテスト");
    msg.addMessage("Hello Static Library");
    
    std::cout << "=== 静的ライブラリ動作確認 ===" << std::endl;
    msg.showAllMessages();
    std::cout << "メッセージ数: " << msg.getNMessages() << std::endl;
    
    return 0;
}