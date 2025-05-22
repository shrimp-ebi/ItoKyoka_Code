/* 機能の実際の動き */

#include <cstring>    // 文字をコピーするため
#include "Message.h"

/** クラスの実装：コンストラクタ
 * クラスのインスタンスが作成されたときに自動的に呼び出される関数
 * メンバ変数の初期化などを行う
 * 名前はクラス名と同じにする
 * Messageクラスの関数であることを明示 
 */
// 機能１：空っぽの箱を作る
Message:: Message() {
    message = nullptr;  // 最初は何も入っていない
}


// 機能２：最初から文字列が入った箱を作る
Message::Message(const char* msg) {
    message = nullptr;

    if (msg != nullptr ) {
        int length = strlen(msg);  // 文字列の長さを測る
        message = new char[length + 1];  // 必要な文だけメモリを確保
        strcpy (message, msg);  // 文字をコピーする
    }
}


/** クラスの実装：デストラクタ
 * クラスのインスタンスが使われなくなったときに自動的に呼び出される関数
 * メモリ領域を解放するなどの処理を行うための関数
 * デストラクタの名前は～クラス名とする
 */

// 機能３：箱を片付ける（メモリを返す）
Message::~Message() {
    if (message != nullptr) {
        delete[] message;  // メモリ開放
        message = nullptr;
    }
}


// 機能４：箱に文字列を入れる
void Message::setMessage(const char* msg) {
    // 古い文字列があれば片付ける
     if (message != nullptr) {
        delete[] message;  // メモリ開放
        message = nullptr;
    }

    // 新しい文字列を入れる
    if (msg != nullptr) {
        message = new char[strlen(msg) + 1]; // 新しい場所を確保
        strcpy(message, msg);  // 文字をコピー
    }
}


// 機能５：箱から文字列を取り出す
char* Message::getMessage(void) const{
    return message;  // 保存してある文字列を返す
}