# Electronic Appendix

This electronic appendix contains the Python scripts and data files that were used for the
main numerical results in the thesis.


# Folder structure

electronic_Appendix/
│
├── pictures_chapter_2/
│   ├── pelve_pictures_blau.py
│   │   → Chapter 2, Figure 2.2:
│   │     
│   │
│   └── pelve_pictures_eps.py
│       → Chapter 2, Figure 2.2:
│         
│
├── asymptotic_normality/
│   ├── asymptotic_normality_iid.py
│   │   → Chapter 3, Figures 3.1 and 3.2:
│   │     
│   │
│   ├── asymptotic_normality_alpha_MA.py
│   │   → Chapter 3, Figure 3.3:
│   │     
│   │
│   └── asymptotic_normality_alpha_AR.py
│       → Chapter 3, Figure 3.4:
│         
│
├── pelve_index/
│   ├── data/
│   │   ├── sp500_prices.csv
│   │   └── 3MY.csv
│   │
│   ├── pelve_bands.py
│   │   → Chapter 4, Figures 4.1 and 4.2:
│   │     
│   │
│   ├── eps_analysis.py
│   │   → Chapter 4, Figure 4.4:
│   │     
│   │
│   └── pelve_median_stock_index.py
│       → Chapter 4:
│         
│
├── treasuries/
│   ├── data/
│   │   └── 3MY.csv
│   │
│   └── historical_treasuries.py
│       → Chapter 5, Figure 5.7:
│         
│
├── backtesting/
│   ├── data/
│   │   └── 3MY.csv
│   │
│   ├── analytical_size_recalibrate_ziegel_grid.py
│   │   → Chapter 5, Figure 5.1:
│   │     
│   │
│   ├── analytical_power_grid.py
│   │   → Chapter 5, Figure 5.2:
│   │     
│   │
│   ├── empirical_size_grid.py
│   │   → Chapter 5:
│   │     
│   │
│   ├── empirical_power_grid.py
│   │   → Chapter 5:
│   │     
│   │
│   ├── garch_t_test_grid.py
│   │   → Chapter 5, Figure 5.6:
│   │     
│   │
│   └── garch_historical_backtesting.py
│       → Chapter 5, Tables 5.1 and 5.2:
│         
│
└── plot_generator/
    ├── boxplot.py
    │   → auxiliary plotting script for grouped rejection-rate boxplots
    │
    ├── boxplot_2.py
    │   → auxiliary plotting script for GARCH rejection-rate boxplots
    │
    └── scatter_plot.py
        → auxiliary plotting script for power-comparison plots