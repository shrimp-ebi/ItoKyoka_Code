// ソース 2-2
// Quitというボタンを配置して、クリックするとプログラムを終了するGTKアプリケーション
#include <gtk/gtk.h>

// コールバック関数の宣言
static void
on_quit (GtkWidget* button, gpointer* user_data){
    // アプリケーションを終了
    // user_dataはGApplicationクラスの変数を指すアドレス
    g_application_quit(G_APPLICATION(user_data));
}

static void
on_activate (GApplication *app, gpointer* user_data){
    GtkWidget *window = gtk_application_window_new (GTK_APPLICATION (app));
    gtk_window_set_title(GTK_WINDOW(window), "Image Viewer");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 300);

    // buttonウィジェット（Quit）作成
    GtkWidget *button = gtk_button_new_with_label("Quit");
    // clickedシグナルにon_quitコールバック関数を接続
    // クリックイベント発生時、clickedシグナルが発生し、on_quitコールバック関数が呼び出される
    g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(on_quit), app);
    // buttonウィジェットをwindowウィジェット（コンテナ）に配置
    // コンテナ：自分自身に他のウィジェットを一つ配置できるウィジェット
    gtk_window_set_child(GTK_WINDOW(window), button);

    gtk_window_present(GTK_WINDOW(window));
}

int main (int argc, char *argv[]){
    GtkApplication *app = gtk_application_new ("org.gtk.tutorial", G_APPLICATION_FLAGS_NONE);
    g_signal_connect (G_OBJECT (app), "activate", G_CALLBACK (on_activate), NULL);
    g_application_run (G_APPLICATION (app), argc, argv);

    return 0;
}