import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def draw_history(history, title):
    data = pd.DataFrame({'Episode': range(1, len(history) + 1), title: history})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y=title, data=data)

    plt.title(title + ' Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()

    plt.show()
