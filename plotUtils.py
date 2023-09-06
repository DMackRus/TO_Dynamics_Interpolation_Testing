import matplotlib.pyplot as plt 

                    #Dark blue, dark red, dark green, light blue, dark red, light green
DIVERGENT_COLORS = ["#0000FF",  "#FF0000",  "#008000", "#3399FF", "#FF6666",  "#66FF66"]
    


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

def linegraph(data, ax, line_color, yAxisTitle, xAxisTitle, xTicks, labels, logyAxis = False):
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

    plt.legend()

    return ax

def stackedBarGraph(data, ax, colors, yAxisTitle, xAxisTitle, xTicks, labels, logyAxis = False):
    normalPosterColour = "#103755"
    highlightPosterColor = "#EEF30D"

    ax.set_xlabel(xAxisTitle)
    ax.set_ylabel(yAxisTitle)

    values = [0] * len(data[0])
    for i in range(len(data)):
        ax.bar(xTicks, data[i], bottom = values, color = colors[i], label = labels[i])
        values += data[i]

    xticks = []
    for i in range(len(xTicks)):
        xticks.append(i)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xTicks, fontsize=11)

    if logyAxis:
        ax.set_yscale('log')

    plt.legend()

    return ax

    # categories = ['Category 1', 'Category 2', 'Category 3']
    # values1 = [10, 15, 5]
    # values2 = [5, 10, 15]
    # values3 = [8, 6, 12]

    # # Calculate the positions for bars
    # bar_width = 0.35
    # index = range(len(categories))

    # # Create the first set of bars
    # plt.bar(index, values1, width=bar_width, label='Value 1')

    # # Create the second set of bars and stack them on top of the first
    # plt.bar(index, values2, width=bar_width, label='Value 2', bottom=values1)

    # # Create the third set of bars and stack them on top of the previous two
    # plt.bar(index, values3, width=bar_width, label='Value 3', bottom=[values1[i] + values2[i] for i in index])

    # # Customize the plot
    # plt.xlabel('Categories')
    # plt.ylabel('Values')
    # plt.title('Stacked Bar Plot')
    # plt.xticks(index, categories)
    # plt.legend()

    # # Show the plot
    # plt.show()