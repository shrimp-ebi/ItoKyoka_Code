// ソース 2-7
// Openメニューで画像を開くことができるGTKアプリケーション
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
on_startup (GApplication *app, gpointer *user_data){
    GtkBuilder *builder = gtk_builder_new_from_file("menu.ui");
    gtk_application_set_menubar(GTK_APPLICATION(app), G_MENU_MODEL(gtk_builder_get_object(builder, "appmenu")));
}

// 画像表示をまとめている関数
// 戻り値：windowウィジェット
static GtkWidget *
image_window_new (GApplication *app){
    GtkWidget *window = gtk_application_window_new(GTK_APPLICATION(app));
    gtk_window_set_title(GTK_WINDOW(window), "Image Viewer");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 300);

    GtkWidget *scrolled_window = gtk_scrolled_window_new();
    gtk_widget_set_vexpand(scrolled_window, true);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled_window), GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_window_set_child(GTK_WINDOW(window), scrolled_window);

    GtkWidget *image = gtk_image_new();
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled_window), image);
    gtk_widget_set_hexpand(image, false);
    gtk_widget_set_vexpand(image, false);
    gtk_widget_set_halign(image, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(image, GTK_ALIGN_CENTER);
    // appにimageウィジェットを"image"という名前で登録
    g_object_set_data(G_OBJECT(app), "image", image);

    return window;
}

// コールバック関数の宣言
// objectはdialogウィジェット（ファイル選択）
static void
open_image (GObject *object, GAsyncResult *result, gpointer user_data){
    // 選択したファイルを取得
    GFile* file = gtk_file_dialog_open_finish(GTK_FILE_DIALOG(object), result, NULL);
    if (file) {
        // user_data（app）に登録した"image"という名前のimageウィジェットを取得
        GtkImage *image = GTK_IMAGE(g_object_get_data(G_OBJECT(user_data), "image"));
        // ファイルのパスを取得
        char *filename = g_file_get_path(file);
        set_image(image, filename);
    }
}

// コールバック関数の宣言
static void
on_menu_open (GSimpleAction *action, GVariant *parameter, gpointer user_data){    
    // dialogウィジェット（ファイル選択）作成
    GtkFileDialog *dialog = gtk_file_dialog_new();
    // タイトルセット
    gtk_file_dialog_set_title(dialog, "Open an Image");
    // 現在の実行フォルダのパスを取得
    // __FILE__はこのソースコード（image-viewer-6.c）のパスを指す
    GFile *dirname = g_file_new_for_path(g_path_get_dirname(__FILE__));
    // ファイル選択ダイアログの初期表示フォルダを現在のパスに設定
    gtk_file_dialog_set_initial_folder(dialog, dirname);
    // ファイル選択ダイアログを表示
    // 第2引数はアプリケーションに対応するウィンドウ（ダイアログ表示時に非活性化するウィンドウ）を指定
    // 第3引数はキャンセル操作を行った時のオブジェクト（今回はNULL）
    // 第4引数はダイアログを閉じた後に呼び出されるコールバック関数を指定
    // 第5引数はその関数に渡されるデータを指定
    gtk_file_dialog_open(dialog, gtk_application_get_active_window(GTK_APPLICATION(user_data)),
                         NULL, open_image, user_data);
}

static void
on_menu_quit (GSimpleAction *action, GVariant *parameter, gpointer user_data){
    g_application_quit(G_APPLICATION(user_data));
}

static GActionEntry app_entries[] = {
    { "menu_open", on_menu_open, NULL, NULL, NULL },
    { "menu_quit", on_menu_quit, NULL, NULL, NULL }
};

static void
on_activate (GApplication *app, gpointer* user_data){
    // windowウィジェット（画像表示可能）作成
    GtkWidget *window = image_window_new (app);
    gtk_application_window_set_show_menubar(GTK_APPLICATION_WINDOW(window), true);
    GActionGroup *actions = (GActionGroup *) g_simple_action_group_new();
    g_action_map_add_action_entries(G_ACTION_MAP(actions), app_entries, G_N_ELEMENTS(app_entries), app);
    gtk_widget_insert_action_group(window, "app", actions);

    gtk_window_present(GTK_WINDOW(window));
}

int main (int argc, char *argv[]){
    GtkApplication *app = gtk_application_new ("org.gtk.tutorial", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect (G_OBJECT (app), "startup", G_CALLBACK (on_startup), NULL);
    g_signal_connect (G_OBJECT (app), "activate", G_CALLBACK (on_activate), NULL);
    g_application_run (G_APPLICATION (app), argc, argv);

    return 0;
}