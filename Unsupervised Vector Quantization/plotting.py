from matplotlib import pyplot as plt


def plot_curve(data, title="", x_lab="", y_lab="", ticks=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    if ticks is not None:
        plt.xticks(ticks)
    plt.plot(range(1, len(data) + 1), data)


def plot_data_points(df, prototypes, prototype_trace, title="", x_lab="", y_lab=""):
    color_zero = "gray"
    color_one = "blue"
    edge_color = (0, 0, 0, 0.5)
    trace_color = (0, 0, 0, 0.85)
    plt.figure()
    plt.scatter(df['X'], df['Y'], label='Class 1', color=color_zero, edgecolors=edge_color)
    for idx, prototype in prototypes.iterrows():
        plt.plot(prototype_trace.iloc[idx]['trace_X'], prototype_trace.iloc[idx]['trace_Y'], color=trace_color)
        plt.scatter(prototype['X'], prototype['Y'], marker='*', color=color_one, s=600, edgecolors=edge_color)
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
