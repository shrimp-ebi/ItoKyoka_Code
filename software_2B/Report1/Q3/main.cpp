#include <iostream>
#include "RepeatMessage.h"

int main (int argc, char *argv[]) {
    RepeatMessage obj(3);

    std::cout << "Input  message: ";
    std::cin >> obj;
    std::cout << "Output message:" << std::endl;
    std::cout << obj;
    
    return 0;
}