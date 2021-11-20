from matplotlib import pyplot as plt


def plot_learning_curve(training_errors):
    plt.title(f"Learning Curve after {len(training_errors) + 1} epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training Error")
    plt.plot(range(1, len(training_errors) + 1), training_errors)
    plt.show()


def plot_data_points(df, prototypes, prototype_trace):
    color_zero = "red"
    color_one = "green"
    df_zero = df[df['Label'] == 0]
    df_one = df[df['Label'] == 1]
    plt.scatter(df_zero['X'], df_zero['Y'], label='Class 1', color=color_zero)
    plt.scatter(df_one['X'], df_one['Y'], label='Class 2', color=color_one)
    for idx, prototype in prototypes.iterrows():
        plt.plot(prototype_trace.iloc[idx]['trace_X'], prototype_trace.iloc[idx]['trace_Y'], color='black')
        color = color_zero if prototype['Label'] == 0 else color_one
        plt.scatter(prototype['X'], prototype['Y'], marker='*', color=color, s=600, edgecolors='black')
    plt.show()
