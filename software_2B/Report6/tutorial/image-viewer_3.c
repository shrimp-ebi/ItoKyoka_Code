// ソース 2-3
// 画像を表示するGTKアプリケーション
// Quitボタンでアプリケーションを終了できる
#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

static void
on_quit (GtkWidget *button, gpointer *user_data){
    g_application_quit(G_APPLICATION(user_data));
}

static void
set_image(GtkImage *image, char *filename){
    if (filename) {
        // pixbuf（画像を扱うクラス）をファイルから読み込む
        GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file(filename, NULL);
        // 画像の幅と高さを取得
        int w = gdk_pixbuf_get_width(pixbuf);
        int h = gdk_pixbuf_get_height(pixbuf);
        // pixbufをimageウィジェット（画像表示）にセット
        gtk_image_set_from_pixbuf(image, pixbuf);
        // imageウィジェットのサイズを設定
        gtk_widget_set_size_request(GTK_WIDGET(image), w, h);
    }
}

static void
on_activate (GApplication *app, gpointer* user_data){
    GtkWidget *window = gtk_application_window_new (GTK_APPLICATION (app));
    gtk_window_set_title(GTK_WINDOW(window), "Image Viewer");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 300);

    // boxウィジェット（パッキングボックス）作成
    // パッキングボックス：自分自身に他のウィジェットを複数配置できるウィジェット
    // GTK_ORIENTATION_VERTICALは子ウィジェットを垂直方向に並べる
    // 5は子ウィジェット間のスペース(5px)
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    // boxウィジェット（パッキングボックス）をwindowウィジェット（コンテナ）に配置
    gtk_window_set_child(GTK_WINDOW(window), box);

    // imageウィジェット（画像表示）作成
    GtkWidget *image = gtk_image_new();
    // imageウィジェット（画像表示）をboxウィジェット（パッキングボックス）に配置
    gtk_box_append(GTK_BOX(box), image);
    // imageウィジェットに画像をセット
    set_image(GTK_IMAGE(image), "sainu(RhinoceroShiba).png");

    GtkWidget *button = gtk_button_new_with_label("Quit");
    g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(on_quit), app);
    // buttonウィジェット（画像表示）をboxウィジェット（パッキングボックス）に配置
    // ※パッキングボックスへの配置は実行順に配置されるため、buttonはimageの下に配置される
    gtk_box_append(GTK_BOX(box), button);

    gtk_window_present(GTK_WINDOW(window));
}

int main (int argc, char *argv[]){
    GtkApplication *app = gtk_application_new ("org.gtk.tutorial", G_APPLICATION_FLAGS_NONE);
    g_signal_connect (G_OBJECT (app), "activate", G_CALLBACK (on_activate), NULL);
    g_application_run (G_APPLICATION (app), argc, argv);

    return 0;
}