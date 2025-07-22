// ソース 2-4
// ウィンドウよりも大きい画像を指定した場合、スクロールバーが表示されるGTKアプリケーション
// ウィンドウのサイズも指定通りにできる
#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

static void
on_quit (GtkWidget *button, gpointer *user_data){
    g_application_quit(G_APPLICATION(user_data));
}

static void
set_image(GtkImage *image, char *filename){
    if (filename) {
        GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file(filename, NULL);
        int w = gdk_pixbuf_get_width(pixbuf);
        int h = gdk_pixbuf_get_height(pixbuf);
        gtk_image_set_from_pixbuf(image, pixbuf);
        gtk_widget_set_size_request(GTK_WIDGET(image), w, h);
    }
}

static void
on_activate (GApplication *app, gpointer* user_data){
    GtkWidget *window = gtk_application_window_new (GTK_APPLICATION (app));
    gtk_window_set_title(GTK_WINDOW(window), "Image Viewer");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 300);

    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_window_set_child(GTK_WINDOW(window), box);
    
    // scrolled_windowウィジェット（スクロールバーのついたウィンドウ）作成
    GtkWidget *scrolled_window = gtk_scrolled_window_new();
    // 親コンテナ（コンテナやボックス）に空きスペースがある場合、縦方向に拡張する
    gtk_widget_set_vexpand(scrolled_window, true);
    // スクロールバーの表示ルールを設定
    // GTK_POLICY_AUTOMATICは必要な時だけ自動でスクロールバーを表示する
    // gtk_scrolled_window_set_policy(window, 水平方向の表示ルール, 垂直方向の表示ルール)
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled_window), GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    // scrolled_windowウィジェットをboxウィジェット（パッキングボックス）に配置
    gtk_box_append(GTK_BOX(box), scrolled_window);

    GtkWidget *image = gtk_image_new();
    // imageウィジェット（画像表示）をscrolled_windowウィジェット（コンテナ）に配置
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled_window), image);
    // imageウィジェットの拡張設定
    // 水平方向、垂直方向、共に拡張しない
    gtk_widget_set_hexpand(image, false);
    gtk_widget_set_vexpand(image, false);
    // 水平方向、垂直方向の配置を中央に設定
    gtk_widget_set_halign(image, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(image, GTK_ALIGN_CENTER);
    set_image(GTK_IMAGE(image), "sainu(RhinoceroShiba).png");

    GtkWidget *button = gtk_button_new_with_label("Quit");
    g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(on_quit), app);
    gtk_box_append(GTK_BOX(box), button);

    gtk_window_present(GTK_WINDOW(window));
}

int main (int argc, char *argv[]){
    GtkApplication *app = gtk_application_new ("org.gtk.tutorial", G_APPLICATION_FLAGS_NONE);
    g_signal_connect (G_OBJECT (app), "activate", G_CALLBACK (on_activate), NULL);
    g_application_run (G_APPLICATION (app), argc, argv);

    return 0;
}