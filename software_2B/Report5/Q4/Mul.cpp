#include "plugin.hpp"
#include <memory>

// Mulクラス: PluginInterfaceを継承
class Mul : public PluginInterface {
public:
    Mul() {}
    ~Mul() {}
    
    std::string getPluginName(void) override {
        return "Mul";
    }
    
    double exec(double a, double b) override {
        return a * b;
    }
};

// クラスのインスタンスを作成する関数
extern "C" {
    std::unique_ptr<PluginInterface> new_instance(void) {
        return std::unique_ptr<PluginInterface>(new Mul);
    }
}