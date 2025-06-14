#include <stdlib.h>
#include <string.h>
#include <string>
#include "Message.h"

Message::Message() : message(nullptr) {
}

Message::Message(const char* _message) {
    message = new char[strlen(_message) + 1];
    strcpy(message, _message);
}

Message::~Message() {
    if (message != nullptr) delete[] message;
}

void Message::setMessage(const char* _message) {
    if (message != nullptr) delete[] message;
    message = new char[strlen(_message) + 1];
    strcpy(message, _message);
}

char* Message::getMessage(void) {
    return message;
}

std::istream& operator>>(std::istream& stream, Message& obj) {
    std::string buffer;
    std::getline(stream, buffer);
    obj.setMessage(buffer.c_str());
    return stream;
}

std::ostream& operator<<(std::ostream& stream, Message& obj) {
    stream << obj.getMessage();
    return stream;
}
