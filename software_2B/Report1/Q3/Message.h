/* Messageクラスの設計図 */ 

#include <iostream> 

class Message {
    private:  
        char* message;  

    public:
        Message();                  
        Message(const char* msg);  
        ~Message();                

        void setMessage(const char* msg);  
        char* getMessage(void) const;           


        // >> 入力演算子：キーボードから文字列を入力してMessageに保存
        friend std::istream& operator>>(std::istream& input, Message& obj);
        // << 出力演算子：Messageの中身を画面に表示
        friend std::ostream& operator<<(std::ostream& output, Message& obj);      
};

// // ２．クラス定義外で宣言
// std::istream& operator>>(std::istream& input, Message& obj);
// std::ostream& operator<<(std::ostream& output, const Message& obj);
