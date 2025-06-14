#include "plugin.hpp"
#include <memory>

// Subクラス: PluginInterfaceを継承
class Sub : public PluginInterface {
public:
    Sub() {}
    ~Sub() {}
    
    std::string getPluginName(void) override {
        return "Sub";
    }
    
    double exec(double a, double b) override {
        return a - b;
    }
};

// クラスのインスタンスを作成する関数
extern "C" {
    std::unique_ptr<PluginInterface> new_instance(void) {
        return std::unique_ptr<PluginInterface>(new Sub);
    }
}