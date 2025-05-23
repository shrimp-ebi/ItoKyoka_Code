/* Messageクラスの設計図 */ 

class Message {
    private:
        char* message;  // ←メンバ変数：メッセージの内容（ポインタ）

    public:
        Message();                  // デフォルトコンストラクタ（文字がないときの初期設定）
        Message(const char* msg);   // 引数ありコンストラクタ（最初から文字をもらうとき）
        ~Message();                 // デストラクタ（いらなくなったときのお片付け）

        void setMessage(const char* msg);  // ←メンバ関数：文字を入れる
        char* getMessage(void) const;            // ←メンバ関数：文字を取り出す
};