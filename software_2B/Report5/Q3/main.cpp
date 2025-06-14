#include <iostream>
#include <dlfcn.h>
#include <memory>
#include "plugin.hpp"

int main() {
    std::cout << "=== 課題3 クラスの動的ロード 動作確認 ===" << std::endl;
    
    // 1. 動的ライブラリをロード
    void* handle = dlopen("./libAdd.so", RTLD_LAZY);
    if (handle == nullptr) {
        std::cerr << "エラー: " << dlerror() << std::endl;
        return 1;
    }
    
    // 2. 関数new_instanceを使えるようにする
    auto new_instance = 
        reinterpret_cast<std::unique_ptr<PluginInterface>(*)(void)>(dlsym(handle, "new_instance"));
    
    if (new_instance == nullptr) {
        std::cerr << "エラー: new_instance関数が見つかりません" << std::endl;
        dlclose(handle);
        return 1;
    }
    
    // 3. 関数new_instanceを使ってクラスのインスタンスを生成して使用
    {
        auto instance = new_instance();
        
        // 4. 動作確認
        std::cout << "プラグイン名: " << instance->getPluginName() << std::endl;
        
        double result1 = instance->exec(10.5, 3.2);
        double result2 = instance->exec(-5.0, 8.7);
        double result3 = instance->exec(0.0, 100.0);
        
        std::cout << "exec(10.5, 3.2) = " << result1 << std::endl;
        std::cout << "exec(-5.0, 8.7) = " << result2 << std::endl;
        std::cout << "exec(0.0, 100.0) = " << result3 << std::endl;
        
        // instanceはここで自動的に破棄される（ライブラリアンロード前）
    }
    
    // 5. ライブラリをアンロード
    dlclose(handle);
    
    return 0;
}