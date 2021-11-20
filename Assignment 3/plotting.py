from matplotlib import pyplot as plt


def plot_curve(train, test, data_idx, x_data, ticks=None, title="", x_lab="", y_lab=""):
    plt.figure()
    color_train = "red"
    color_test = "green"
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    if ticks is not None:
        plt.xticks(ticks)
    plt.plot(x_data, [tup[data_idx] for tup in train], color=color_train, label="Training Error")
    plt.plot(x_data, [tup[data_idx] for tup in test], color=color_test, label="Testing Error")
    plt.legend()
