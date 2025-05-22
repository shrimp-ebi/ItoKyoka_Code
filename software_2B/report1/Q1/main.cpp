/* 作ったMessageクラスをテストする */

#include <iostream>
#include "Message.h"

int main (int argc, char *argv[]) {
    Message obj;

    obj.setMessage("Hello World.");  // 機能４

    std::cout << obj.getMessage() << std::endl;  // 機能５を出力

    return 0;
}