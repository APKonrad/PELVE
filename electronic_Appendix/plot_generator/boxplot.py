import matplotlib.pyplot as plt

# ============================================================
# DATA
# ============================================================

mc_levels = [100, 1000, 2500, 5000]
# Example Values
pelve = {
    100:  [0.0700,0.0500,0.0600,0.0500,0.0200,0.0200,0.0300,0.0400,0.0700,0.0200,
           0.0300,0.0300,0.0900,0.0800,0.0900,0.0700,0.0400,0.0700,0.0200,0.0700],
    1000: [0.0490,0.0620,0.0400,0.0460,0.0460,0.0500,0.0420,0.0450,0.0520,0.0440,
           0.0560,0.0430,0.0480,0.0400,0.0360,0.0580,0.0420,0.0530,0.0460,0.0520],
    2500: [0.0404,0.0532,0.0424,0.0464,0.0512,0.0568,0.0444,0.0456,0.0480,0.0432,
           0.0420,0.0496,0.0500,0.0452,0.0468,0.0420,0.0436,0.0464,0.0532,0.0452],
    5000: [0.0474,0.0498,0.0482,0.0374,0.0452,0.0484,0.0486,0.0494,0.0490,0.0518,
           0.0458,0.0460,0.0438,0.0534,0.0488,0.0458,0.0420,0.0468,0.0476,0.0454],
}

ziegel_chi2 = { #MC
    100:  [0.0800,0.1200,0.0600,0.0500,0.0500,0.0500,0.0700,0.0800,0.0400,0.0300,
           0.0800,0.0800,0.0500,0.0800,0.0900,0.0600,0.0700,0.0800,0.0100,0.0500],
    1000: [0.0490,0.0620,0.0600,0.0760,0.0490,0.0660,0.0630,0.0600,0.0570,0.0590,
           0.0560,0.0570,0.0630,0.0470,0.0480,0.0660,0.0500,0.0600,0.0660,0.0560],
    2500: [0.0548,0.0612,0.0540,0.0588,0.0540,0.0592,0.0588,0.0652,0.0608,0.0496,
           0.0540,0.0596,0.0616,0.0556,0.0624,0.0624,0.0464,0.0608,0.0536,0.0524],
    5000: [0.0624,0.0508,0.0554,0.0582,0.0582,0.0588,0.0584,0.0570,0.0596,0.0530,
           0.0596,0.0568,0.0568,0.0572,0.0630,0.0604,0.0608,0.0540,0.0550,0.0538],
}



def plot_grouped_boxplots(
    data_left,
    data_right,
    mc_levels,
    label_left="PELVE",
    label_right="ZIEGEL chi²",
    ref_line=0.05,
    output_path=None
):



    positions_left = [1.0, 4.0, 7.0, 10.0]
    positions_right = [1.8, 4.8, 7.8, 10.8]
    centers = [(a + b) / 2 for a, b in zip(positions_left, positions_right)]

    #fig, ax = plt.subplots(figsize=(11, 6.7), dpi=160)
    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=150)

    common_kwargs = dict(
        widths=0.55,
        patch_artist=False,
        showfliers=False,   # keine Outlier-Punkte
        medianprops=dict(linewidth=1.3),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    bp1 = ax.boxplot(
        [data_left[m] for m in mc_levels],
        positions=positions_left,
        **common_kwargs
    )
    bp2 = ax.boxplot(
        [data_right[m] for m in mc_levels],
        positions=positions_right,
        **common_kwargs
    )


    ax.axhline(ref_line, linewidth=1.2)

    ax.set_xticks(centers)
    ax.set_xticklabels([str(m) for m in mc_levels])
    ax.set_xlabel("Monte Carlo iterations")
    ax.set_ylabel("Rejection rate")

    # Achsenbereich bei Bedarf anpassen
    ax.set_xlim(0.2, 11.6)
    ax.set_ylim(0.0, 0.13)

    # optionale kleine Legende über Dummy-Linien
    ax.plot([], [], label=label_left)
    ax.plot([], [], label=label_right)
    #ax.legend(frameon=False)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()




plot_grouped_boxplots(
    data_left=pelve,
    data_right=ziegel_chi2,
    mc_levels=mc_levels,
    label_left="PELVE",
    label_right="ZIEGEL chi²",
    ref_line=0.05,
    output_path="boxplot_pelve_ziegel_chi2.png"
)