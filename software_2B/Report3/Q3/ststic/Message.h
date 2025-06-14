#include <string>
#include <vector>

class Message {
private:
    std::vector<std::string> message;

public:
    Message();
    Message(const std::string& message_string);
    Message(const std::vector<std::string>& message_vector);
    ~Message();

    void addMessage(const std::string& message_string);
    std::string getMessage(int message_id);
    void showAllMessages(void);
    int getNMessages(void);
};