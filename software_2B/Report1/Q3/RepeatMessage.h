/* RepeatMessageクラスの設計図 - Messageクラスを継承 */

#include <iostream>
#include "Message.h"  // 親クラスのヘッダーファイル

/** クラスの継承
 * 既存クラスの機能を実装し直すことなく、既存クラスにない機能だけ実装するだけで済む
 * 継承元のクラスのメンバ変数やメンバ関数を継承先のクラスでも使用することができる
 * 継承元のメンバ変数にアクセスしたい場合、”protected”に設定する（privateでは継承したクラスからも参照できない）
 * 継承元のクラス名を書く　　↓
*/
class RepeatMessage : public Message {  
    private:
        int nloops;  // 繰り返し回数を保存するメンバ変数

    public:
        RepeatMessage();  // デフォルトコンストラクタ
        RepeatMessage(int num);  // 繰り返し回数を指定するコンストラクタ
        ~RepeatMessage();  // デストラクタ

        // << 出力演算子のオーバーロード（RepeatMessage専用）
        friend std::ostream& operator<< (std::ostream& output, const RepeatMessage& obj);
};