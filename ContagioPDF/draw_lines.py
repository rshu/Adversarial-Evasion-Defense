import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == "__main__":
    df = pd.read_csv('distance_result.csv')
    print(df.shape)

    df['distance'] = df['distance'].round(3)
    df['FGSM'] = df['FGSM'].round(4)

    distance = df['distance'].to_list()
    fgsm = df['FGSM'].to_list()

    print(distance)
    print(fgsm)

    plt.scatter(distance, fgsm)
    # plt.plot(distance, fgsm)

    plt.annotate("Best Performance Point", (df.iloc[0]['distance'], df.iloc[0]['FGSM']), xycoords='data',
                 xytext=(0.5, 0.6), arrowprops=dict(arrowstyle='->'))

    plt.xlabel("distance")
    plt.ylabel('FGSM Accuracy')
    plt.show()
