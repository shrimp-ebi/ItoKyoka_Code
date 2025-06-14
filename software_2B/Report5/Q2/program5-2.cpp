#include <iostream>
#include <dlfcn.h>

int main() {
    std::cout << "=== 課題2 動的ロード 動作確認 ===" << std::endl;
    
    // 1. ヘッダファイル dlfcn.h をインクルード済み
    
    // 2. 関数 dlopen でライブラリをロード
    void* handle = dlopen("./libcalc.so", RTLD_LAZY);
    if (handle == nullptr) {
        std::cerr << "エラー: " << dlerror() << std::endl;
        return 1;
    }
    
    // 3. 関数 dlsym を用いて関数へのポインタを取得
    double (*my_add_func)(double, double) = 
        (double(*)(double, double)) dlsym(handle, "my_add");
    
    double (*my_sub_func)(double, double) = 
        (double(*)(double, double)) dlsym(handle, "my_sub");
    
    // エラーチェック
    if (my_add_func == nullptr || my_sub_func == nullptr) {
        std::cerr << "エラー: 関数が見つかりません" << std::endl;
        dlclose(handle);
        return 1;
    }
    
    // 4. 動作確認
    double a = my_add_func(10, 20);
    double b = my_sub_func(10, 20);
    
    std::cout << "my_add(10, 20) = " << a << std::endl;
    std::cout << "my_sub(10, 20) = " << b << std::endl;
    
    // 5. 使用しなくなったら関数 dlclose を実行
    dlclose(handle);
    
    return 0;
}