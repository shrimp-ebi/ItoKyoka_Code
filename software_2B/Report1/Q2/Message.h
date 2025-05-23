/* Messageクラスの設計図 */ 

#include <iostream>  // 入出力ストリーム

class Message {
    private:
        char* message;  

    public:
        Message();                  
        Message(const char* msg);  
        ~Message();                

        void setMessage(const char* msg);  
        char* getMessage(void) const;           


        /** === 課題2の新しい実装 ===
         * 演算子オーバーロードの宣言
         * 演算子を独自クラス用に定義しなおすことができる
         * クラスのメンバ関数ではない
         */

        // １．クラス定義内でオーバーロード→friend キーワードを付ける
        // >> 入力演算子：キーボードから文字列を入力してMessageに保存
        friend std::istream& operator>>(std::istream& input, Message& obj);
        // << 出力演算子：Messageの中身を画面に表示
        friend std::ostream& operator<<(std::ostream& output, const Message& obj);      
};

// // ２．クラス定義外で宣言
// std::istream& operator>>(std::istream& input, Message& obj);
// std::ostream& operator<<(std::ostream& output, const Message& obj);
