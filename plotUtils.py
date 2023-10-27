import matplotlib.pyplot as plt
import numpy as np 
import colorsys

                    #Dark blue, dark red, dark green, light blue, purple, light green
DIVERGENT_COLORS = ["#0000FF",  "#FF0000",  "#008000", "#3399FF", "#e303fc",  "#66FF66"]
ROYAL_COLORS = [
    "#4169E1",  # Royal Blue
    "#DC143C",  # Crimson Red
    "#008000",  # Emerald Green
    "#FFD700",  # Golden Yellow
    "#333333",  # Charcoal Gray
    "#9966CC",  # Amethyst Purple
    "#FFA500"   # Tangerine Orange
]


SHADES_OF_BLUE = ["#0000FF", "#4169E1", "#87CEEB"]
SHADES_OF_GREEN = ["#006400", "#008000", "#00FF00"]

def box_plot(data, fill_color, yAxisTitle, ax, labels, logyAxis = False, baseline_yLine = False):
    normalPosterColour = "#103755"
    highlightPosterColor = "#EEF30D"


    bp = ax.boxplot(data, patch_artist=True, meanprops={"marker":"s","markerfacecolor":highlightPosterColor, "markeredgecolor":highlightPosterColor}, showmeans=True, showfliers=False)
    if logyAxis:
        ax.set_yscale('log')
    black = "#1c1b1a"

    for element in ['medians']:
        plt.setp(bp[element], color=black)

    for element in ['means']:
        plt.setp(bp[element], color=highlightPosterColor)
        # element.set(color="#a808c4")

    # for bpMean in bp['means']:
    #     bpMean.set(color="#a808c4")

    # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    #     plt.setp(bp[element], color=black)


    dynamicColor = "#ff8400"
    baselineColor = "#0057c9"
    setIntervalColour = "#9d00c9"
    backgroundColour = "#d6d6d6"

    
    index = 0
    for patch in bp['boxes']:
        patch.set(facecolor=normalPosterColour)

    labelSize = 11

    ax.set_ylabel(yAxisTitle, fontsize=labelSize)

    if(baseline_yLine):
        ax.axhline(y=0, color=baselineColor, linewidth=1.5, alpha=0.5)

    xticks = []
    for i in range(len(labels)):
        xticks.append(i + 1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=11)
        
    return bp   

def linegraph(data, ax, line_color, yAxisTitle, xAxisTitle, xTicks, labels, legend = None, logyAxis = False):
    normalPosterColour = "#103755"
    highlightPosterColor = "#EEF30D"

    ax.set_xlabel(xAxisTitle)
    ax.set_ylabel(yAxisTitle)

    for i in range(len(data)):
        ax.plot(data[i], color = line_color[i], label = labels[i])

    xticks = []
    for i in range(len(xTicks)):
        xticks.append(i)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xTicks, fontsize=11)

    if logyAxis:
        ax.set_yscale('log')

    if(legend != None):
        print(yAxisTitle)
        ax.legend(handles=legend)

    return ax

def stackedBarGraph(data, ax, colors, yAxisTitle, xAxisTitle, xTicks, labels, logyAxis = False):
    normalPosterColour = "#103755"
    highlightPosterColor = "#EEF30D"

    ax.set_xlabel(xAxisTitle)
    ax.set_ylabel(yAxisTitle)

    # data - [time segment, key point method, horizon length]
    width = 0.35  # Width of each bar
    x = np.arange(data.shape[2])
    baseline_colors = SHADES_OF_BLUE
    adaptive_colors = SHADES_OF_GREEN

    values = np.zeros((2, len(data[0, 0])))

    # baseline
    for i in range(len(data)):
        ax.bar(x - width/1.8, data[i, 0], width, bottom = values[0], color = baseline_colors[i], label = labels[i] + " - Baseline")
        # ax.bar(x, data[i, 0], width, bottom = values[0], color = baseline_colors[i], label = labels[i] + " - baseline")
        values[0] += data[i, 0]

    # adaptive jerk
    for i in range(len(data)):
        ax.bar(x + width/1.8, data[i, 1], width, bottom = values[1], color = adaptive_colors[i], label = labels[i] + " - Mag vel change")
        values[1] += data[i, 1]

    xticks = []
    for i in range(len(xTicks)):
        xticks.append(i)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xTicks, fontsize=11)

    if logyAxis:
        ax.set_yscale('log')

    plt.legend()

    return ax

if __name__ == "__main__":
    derivs = np.array([2.2, 5.2, 20])
    bp = np.array([0.05, 0.07, 0.18])
    fp = np.array([0.2, 0.4, 0.7])
    stackedData = np.zeros((3, 1, derivs.shape[0]))
    
    stackedData[0, 0, :] = derivs
    stackedData[1, 0, :] = bp
    stackedData[2, 0, :] = fp

    # print("avg time getting derivs: " + str(avgTimesDerivsOther))
    # stackedData[0, 1, :] = avgTimesDerivsOther
    # stackedData[1, 1, :] = avgTimesBPOther
    # stackedData[2, 1, :] = avgTimesFPOther


    # data = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    ax = plt.subplot(111)
    stackedBarGraph(stackedData, ax, DIVERGENT_COLORS, "Average time per iteration (s)", "Dimensionality of problem", ["9", "15", "23"], ["Derivs", "BP", "FP"])
    plt.show()