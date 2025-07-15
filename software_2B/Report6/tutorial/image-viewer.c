/*
GTKを使ったC言語のGUIアプリケーション
*/

#include <gtk/gtk.h>


//　ウィンドウを作るための関数
static void
on_activate (GApplication* app, gpointer* user_data) {
    // アプリケーション用のウィンドウを作成
    GtkWidget* window = gtk_application_window_new(GTK_APPLICATION(app));  

    // ウィンドウのタイトルと初期サイズを設定
    gtk_window_set_title(GTK_WINDOW(window), "Image Viewer");  
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 300); 
    
    // ウィンドウを表示
    gtk_window_present(GTK_WINDOW(window));  
}

int main (int argc, char *argv[]) { 

    // GTKアプリケーションの作成（org.gtk.tutorialはアプリの識別名）
    GtkApplication* app = gtk_application_new("org.gtk.tutorial", GDK_APPLICATION_FLAGS_NONE); // 新しいGTKアプリケーションを作成
    
    // アプリが「起動された」ときに on_activate を呼ぶようにシグナル（イベント）を接続
    g_signal_connect (G_OBJECT(app), "activate", G_CALLBACK(on_activate), NULL); 
    
    // アプリケーションを実行
    g_application_run (G_APPLICATION(app), argc, argv); 

    return 0
}