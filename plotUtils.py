import matplotlib.pyplot as plt
import numpy as np 
import colorsys

                    #Dark blue, dark red, dark green, light blue, dark red, light green
DIVERGENT_COLORS = ["#0000FF",  "#FF0000",  "#008000", "#3399FF", "#FF6666",  "#66FF66"]

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

    # data - [time segment, key point method, horizon length]
    width = 0.35  # Width of each bar
    x = np.arange(data.shape[2])
    baseline_colors = SHADES_OF_BLUE
    adaptive_colors = SHADES_OF_GREEN

    values = np.zeros((2, len(data[0, 0])))

    # baseline
    for i in range(len(data)):
        ax.bar(x - width/1.8, data[i, 0], width, bottom = values[0], color = baseline_colors[i], label = labels[i] + " - baseline")
        values[0] += data[i, 0]

    # adaptive jerk
    for i in range(len(data)):
        ax.bar(x + width/1.8, data[i, 1], width, bottom = values[1], color = adaptive_colors[i], label = labels[i] + " - adaptive_jerk")
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

def generate_shades_of_hex_color(base_hex_color, num_shades):
    # Convert the base hexadecimal color to RGB format
    base_color = tuple(int(base_hex_color[i:i+2], 16) for i in (1, 3, 5))

    # Convert the base color from RGB to HSV
    base_color_hsv = colorsys.rgb_to_hsv(base_color[0] / 255.0, base_color[1] / 255.0, base_color[2] / 255.0)

    # Generate shades by varying the value (brightness) component, but keep saturation high
    shades = []
    for i in range(num_shades):
        value = max(0.4, base_color_hsv[2] * (1 - i / (num_shades - 1)))  # Vary the value from 0.2 (original) to 1.0 (original)
        rgb_color = colorsys.hsv_to_rgb(base_color_hsv[0], 1.0, value)  # Keep saturation at 1.0
        # Convert RGB values from [0.0, 1.0] to [0, 255] and round them to integers
        rgb_color = tuple(int(val * 255) for val in rgb_color)
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb_color)
        shades.append(hex_color)

    return shades

if __name__ == "__main__":
    darkBlue = (0, 0, 256)
    shades_of_blue = generate_shades_of_color(darkBlue, 3)
    print(shades_of_blue)
    shades_of_blue = rgb_to_hex(shades_of_blue[0])
    print(shades_of_blue)