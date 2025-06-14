#include "plugin.hpp"
#include <memory>

// Addクラス: PluginInterfaceを継承
class Add : public PluginInterface {
public:
    Add() {}
    ~Add() {}
    
    // プラグイン名を返す関数
    std::string getPluginName(void) override {
        return "Add";
    }
    
    // 引数に与えたdouble型の値を足し算した結果を返す関数  
    double exec(double a, double b) override {
        return a + b;
    }
};

// クラスのインスタンスを作成する関数
extern "C" {
    std::unique_ptr<PluginInterface> new_instance(void) {
        return std::unique_ptr<PluginInterface>(new Add);
    }
}