// ソース 2-6
// XML形式のファイル（menu.ui）を読み込み、左上にFileメニューバーを追加したGTKアプリケーション
// Openメニューは未実装
// Quitメニューでアプリケーションを終了できる
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

// コールバック関数の宣言
static void
on_startup (GApplication *app, gpointer *user_data){
    // XML形式のファイル（menu.ui）を読み込む
    GtkBuilder *builder = gtk_builder_new_from_file("menu.ui");
    // 読み込んだXMLからappmenuという名前のメニューバーウィジェットを取得し、アプリケーションに設定
    gtk_application_set_menubar(GTK_APPLICATION(app), G_MENU_MODEL(gtk_builder_get_object(builder, "appmenu")));
}

// 画像表示をまとめている関数
// 次のソースコードで使用する
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
    set_image(GTK_IMAGE(image), "sainu(RhinoceroShiba).png");

    return window;
}

// コールバック関数の宣言
static void
on_menu_open (GSimpleAction *action, GVariant *parameter, gpointer user_data){
    // 「この機能はまだ実装されていない\n」
    printf("This function is not implemented yet.\n");
}

// コールバック関数の宣言
static void
on_menu_quit (GSimpleAction *action, GVariant *parameter, gpointer user_data){
    g_application_quit(G_APPLICATION(user_data));
}

// メニューアイテムに対するコールバック関数の記述
// GActionEntryは構造体
static GActionEntry app_entries[] = {
    // 1. メニューアイテムと関連付けするための文字列
    // 2. メニューアイテムが選択されたときに呼び出されるコールバック関数
    // 3. パラメータタイプ
    // 4. 状態
    // 5. メニューアイテムの状態が変わったときに呼び出されるコールバック関数
    { "menu_open", on_menu_open, NULL, NULL, NULL },
    { "menu_quit", on_menu_quit, NULL, NULL, NULL }
};

static void
on_activate (GApplication *app, gpointer* user_data){
    GtkWidget *window = gtk_application_window_new (app);
    // ウィンドウのメニューバー表示を有効にする
    gtk_application_window_set_show_menubar(GTK_APPLICATION_WINDOW(window), true);
    // アクショングループを作成
    GActionGroup *actions = (GActionGroup *) g_simple_action_group_new();
    // アクションエントリ（app_entries構造体）をアクショングループに追加
    g_action_map_add_action_entries(G_ACTION_MAP(actions), app_entries, G_N_ELEMENTS(app_entries), app);
    // アクショングループをウィンドウに追加
    gtk_widget_insert_action_group(window, "app", actions);
    
    gtk_window_present(GTK_WINDOW(window));
}

int main (int argc, char *argv[]){
    GtkApplication *app = gtk_application_new ("org.gtk.tutorial", G_APPLICATION_FLAGS_NONE);
    // startupシグナルにon_startupコールバック関数を接続
    // コンストラクタ（アプリ起動時に一回だけ）イベント発生時、startupシグナルが発生し、on_startupコールバック関数が呼び出される
    g_signal_connect (G_OBJECT (app), "startup", G_CALLBACK (on_startup), NULL);
    g_signal_connect (G_OBJECT (app), "activate", G_CALLBACK (on_activate), NULL);
    g_application_run (G_APPLICATION (app), argc, argv);

    return 0;
}