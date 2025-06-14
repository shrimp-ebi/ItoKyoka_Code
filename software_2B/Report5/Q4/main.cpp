#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>   // 課題1: ディレクトリ走査
#include <string.h>
#include <dlfcn.h>    // 課題3: 動的ロード
#include <memory>
#include "plugin.hpp"

int main() {
    std::cout << "=== 課題4 プラグインシステム 動作確認 ===" << std::endl;
    
    // 1. ディレクトリ走査（課題1の技術）
    std::vector<std::string> plugin_files;
    
    DIR* dir = opendir("./plugin");
    if (dir == nullptr) {
        std::cerr << "エラー: pluginディレクトリを開けません" << std::endl;
        return 1;
    }
    
    struct dirent* file;
    while ((file = readdir(dir)) != nullptr) {
        if (strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0) {
            std::string filename = file->d_name;
            // .soファイルのみをフィルタリング
            if (filename.find(".so") != std::string::npos) {
                plugin_files.push_back("./plugin/" + filename);
            }
        }
    }
    closedir(dir);
    
    std::cout << "発見されたプラグイン数: " << plugin_files.size() << std::endl;
    for (const auto& plugin : plugin_files) {
        std::cout << "  " << plugin << std::endl;
    }
    std::cout << std::endl;
    
    // 2. 各プラグインを動的ロードして使用（課題3の技術）
    std::vector<void*> handles;  // dlcloseのためにハンドルを保存
    
    for (const auto& plugin_path : plugin_files) {
        std::cout << "--- " << plugin_path << " ---" << std::endl;
        
        // 動的ロード
        void* handle = dlopen(plugin_path.c_str(), RTLD_LAZY);
        if (handle == nullptr) {
            std::cerr << "エラー: " << dlerror() << std::endl;
            continue;
        }
        handles.push_back(handle);
        
        // new_instance関数を取得
        auto new_instance = 
            reinterpret_cast<std::unique_ptr<PluginInterface>(*)(void)>(dlsym(handle, "new_instance"));
        
        if (new_instance == nullptr) {
            std::cerr << "エラー: new_instance関数が見つかりません" << std::endl;
            continue;
        }
        
        // インスタンス作成と動作確認
        {
            auto instance = new_instance();
            
            std::cout << "プラグイン名: " << instance->getPluginName() << std::endl;
            
            // テストケース実行
            double test_cases[][2] = {{10.0, 5.0}, {8.5, 2.5}, {-3.0, 4.0}};
            
            for (auto& test : test_cases) {
                double result = instance->exec(test[0], test[1]);
                std::cout << "  exec(" << test[0] << ", " << test[1] << ") = " << result << std::endl;
            }
        }
        std::cout << std::endl;
    }
    
    // 3. すべてのライブラリをアンロード
    for (void* handle : handles) {
        dlclose(handle);
    }
    
    std::cout << "=== プラグインシステム完了 ===" << std::endl;
    return 0;
}