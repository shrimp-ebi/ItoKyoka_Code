/* 機能の実際の動き */

#include <iostream>
#include <cstring>    
#include "Message.h"

// 機能１：空っぽの箱を作る【コンストラクタ】
Message:: Message() {
    message = nullptr; 
}

// 機能２：最初から文字列が入った箱を作る【引数ありコンストラクタ】
Message::Message(const char* msg) {
    message = nullptr;

    if (msg != nullptr ) {
        int length = strlen(msg);  
        message = new char[length + 1]; 
        strcpy (message, msg);  
    }
}

// 機能３：箱を片付ける（メモリを返す）【デストラクタ】
Message::~Message() {
    if (message != nullptr) {
        delete[] message;  
        message = nullptr;
    }
}

// 機能４：箱に文字列を入れる【関数】
void Message::setMessage(const char* msg) {
     if (message != nullptr) {
        delete[] message;  // メモリ開放
        message = nullptr;
    }

    if (msg != nullptr) {
        message = new char[strlen(msg) + 1];
        strcpy(message, msg);  
    }
}

// 機能５：箱から文字列を取り出す【関数】
char* Message::getMessage(void) const{
    return message;
}



// >> 入力演算子：キーボードから文字列を入力してMessageに保存
std::istream& operator>>(std::istream& input, Message& obj) {
    char buffer[1000];  

    input.getline(buffer, 1000); 
    obj.setMessage(buffer);  
    return input;
}

// << 出力演算子：Messageの中身を画面に表示
std::ostream& operator<<(std::ostream& output, Message& obj) {
    if (obj.getMessage() != nullptr) {
        output << obj.getMessage();
    } 
    return output;
}