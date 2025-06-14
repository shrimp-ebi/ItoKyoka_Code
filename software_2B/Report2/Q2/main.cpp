#include <iostream>
#include "Message.h"

int main(int argc, char *argv[]) {
    Message obj1("Hello World.");
    Message obj2 = obj1;

    // アドレスを表示
    printf("obj1.message のアドレス: %p\n", obj1.getMessage());
    printf("obj2.message のアドレス: %p\n", obj2.getMessage());

    
    std::cout << obj2 << std::endl;

    return 0;
}