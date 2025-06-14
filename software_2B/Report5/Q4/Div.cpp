#include "plugin.hpp"
#include <memory>

// Divクラス: PluginInterfaceを継承
class Div : public PluginInterface {
public:
    Div() {}
    ~Div() {}
    
    std::string getPluginName(void) override {
        return "Div";
    }
    
    double exec(double a, double b) override {
        if (b == 0.0) {
            return 0.0;  // ゼロ除算対策（簡易版）
        }
        return a / b;
    }
};

// クラスのインスタンスを作成する関数
extern "C" {
    std::unique_ptr<PluginInterface> new_instance(void) {
        return std::unique_ptr<PluginInterface>(new Div);
    }
}