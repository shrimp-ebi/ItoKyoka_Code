#include "RepeatMessage.h"
#include <iostream>
#include <cstring>

RepeatMessage:: RepeatMessage() {
}

RepeatMessage:: RepeatMessage(int num) {
    nloops = num;
}

RepeatMessage:: ~RepeatMessage() {
    // 親クラスのデストラクタが自動的に呼び出されるので、特に何もしない
}

// << 出力演算子
std::ostream& operator<<(std::ostream&output, const RepeatMessage& obj) {
    // メッセージを指定回数繰り返し表示
    for (int i=0; i<obj.nloops; i++) {
        if (obj.getMessage() != nullptr) {
            output << obj.getMessage() << std::endl;
        } 
    }
    return output;
}