# gnuplotスクリプト: plot_script.gp

# 出力設定
set terminal png size 1200,800
set output 'statistics_analysis.png'

# 全体のレイアウト設定
set multiplot layout 2,2 title "Sample Statistics vs Data Size N (μ=5.0, σ=2.0)"

# グラフ1: 標本平均の変化
set xlabel 'Data Size N'
set ylabel 'Sample Mean'
set title 'Sample Mean vs N'
set logscale x
set grid
# 理論値の水平線を追加
set arrow from 10,5.0 to 10000,5.0 nohead lc rgb "red" lw 2
plot 'statistics_vs_N.dat' using 1:2 with linespoints lw 2 pt 7 ps 1 title 'Sample Mean', \
     5.0 with lines lc rgb "red" lw 2 lt 2 title 'Theoretical Mean (μ=5.0)'

# グラフ2: 標本分散の変化  
set xlabel 'Data Size N'
set ylabel 'Sample Variance'
set title 'Sample Variance vs N'
set logscale x
set grid
# 理論値の水平線を追加
set arrow from 10,4.0 to 10000,4.0 nohead lc rgb "red" lw 2
plot 'statistics_vs_N.dat' using 1:3 with linespoints lw 2 pt 7 ps 1 title 'Sample Variance', \
     4.0 with lines lc rgb "red" lw 2 lt 2 title 'Theoretical Variance (σ²=4.0)'

# グラフ3: 平均の誤差
set xlabel 'Data Size N'
set ylabel 'Mean Error |x̄ - μ|'
set title 'Mean Error vs N'
set logscale x
set logscale y
set grid
plot 'statistics_vs_N.dat' using 1:4 with linespoints lw 2 pt 7 ps 1 title 'Mean Error'

# グラフ4: 分散の誤差
set xlabel 'Data Size N'
set ylabel 'Variance Error |s² - σ²|'
set title 'Variance Error vs N'
set logscale x
set logscale y
set grid
plot 'statistics_vs_N.dat' using 1:5 with linespoints lw 2 pt 7 ps 1 title 'Variance Error'

unset multiplot

# 追加: 個別グラフも生成
set terminal png size 800,600

# 標本平均のみのグラフ
set output 'sample_mean_vs_N.png'
set title 'Sample Mean vs Data Size N (μ=5.0, σ=2.0)'
set xlabel 'Data Size N'
set ylabel 'Sample Mean'
set logscale x
unset logscale y
set grid
set yrange [4.5:5.5]
plot 'statistics_vs_N.dat' using 1:2 with linespoints lw 3 pt 7 ps 1.5 lc rgb "blue" title 'Sample Mean', \
     5.0 with lines lc rgb "red" lw 3 lt 2 title 'Theoretical Mean (μ=5.0)'

# 標本分散のみのグラフ  
set output 'sample_variance_vs_N.png'
set title 'Sample Variance vs Data Size N (μ=5.0, σ=2.0)'
set xlabel 'Data Size N'  
set ylabel 'Sample Variance'
set logscale x
unset logscale y
set grid
set yrange [3.0:5.0]
plot 'statistics_vs_N.dat' using 1:3 with linespoints lw 3 pt 7 ps 1.5 lc rgb "green" title 'Sample Variance', \
     4.0 with lines lc rgb "red" lw 3 lt 2 title 'Theoretical Variance (σ²=4.0)'