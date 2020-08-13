import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import matplotlib.cm as cm
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import math

if __name__ == "__main__":
    df = pd.read_csv('distance_result.csv')
    print(df.shape)

    df['distance'] = df['distance'].round(3)
    df['FGSM'] = df['FGSM'].round(4)
    df['BIM-A'] = df['BIM-A'].round(4)
    df['BIM-B'] = df['BIM-B'].round(4)
    df['deepfool'] = df['deepfool'].round(4)
    df['loss'] = df['loss'].round(5)

    distance = df['distance'].to_list()
    loss = df['loss'].to_list()

    # loss_delta = []
    #
    # for i in range(df.shape[0]):
    #     delta = df.iloc[i]['loss'] - df.iloc[0]['loss']
    #     loss_delta.append(delta)
    #
    # print(loss_delta)

    # combination = []
    #
    # w1 = 0.5
    # w2 = 0
    #
    # for i in range(len(distance)):
    #     each = w1 * distance[i] + w2 * loss_delta[i]
    #     combination.append(each)
    #
    # print(combination)

    fgsm = df['FGSM'].to_list()
    bim_a = df['BIM-A'].to_list()
    bim_b = df['BIM-B'].to_list()
    deepfool = df['deepfool'].to_list()

    # # print(fgsm)
    # base_fgsm = fgsm[0]
    # for i in range(len(fgsm)):
    #     fgsm[i] = fgsm[i] - base_fgsm
    #
    # # print(fgsm)
    # base_bim_a = bim_a[0]
    # for i in range(len(bim_a)):
    #     bim_a[i] = bim_a[i] - base_bim_a
    #
    # base_bim_b = bim_b[0]
    # for i in range(len(bim_b)):
    #     bim_b[i] = bim_b[i] - base_bim_b

    # print(distance)
    # print(fgsm)
    plt.subplot(221)
    plt.scatter(distance, fgsm, color='r', alpha=0.3, edgecolors='white', label='FGSM')
    plt.ylim(0.0, 1.1)
    plt.xlabel("Distance")
    plt.ylabel('Accuracy under attacks.')
    plt.legend()
    plt.axhline(y=0.9921, color='r', linestyle='--')
    plt.axhline(y=0.60, color='k', linestyle='--')
    plt.axvline(x=0.50, color='b', linestyle='--')
    # plt.annotate("Best model", (0.0, 0.6), xycoords='data',
    #      xytext=(0.0, 0.8), arrowprops=dict(arrowstyle='->'))

    plt.subplot(222)
    plt.scatter(distance, bim_a, color='b', alpha=0.3, edgecolors='white', label='BIM-A')
    plt.ylim(0.0, 1.1)
    plt.xlabel("Distance")
    plt.ylabel('Accuracy under attacks.')
    plt.legend()
    plt.axhline(y=0.9921, color='r', linestyle='--')
    plt.axhline(y=0.28, color='k', linestyle='--')
    plt.axvline(x=0.50, color='b', linestyle='--')
    # plt.annotate("Best model", (0.0, 0.28), xycoords='data',
    #              xytext=(0.0, 0.8), arrowprops=dict(arrowstyle='->'))

    plt.subplot(223)
    plt.scatter(distance, bim_b, color='g', alpha=0.3, edgecolors='white', label='BIM-B')
    plt.ylim(0.0, 1.1)
    # plt.plot(distance, fgsm)

    # plt.annotate("Best Performance Point", (df.iloc[0]['distance'], df.iloc[0]['FGSM']), xycoords='data',
    #              xytext=(0.5, 0.6), arrowprops=dict(arrowstyle='->'))

    plt.xlabel("Distance")
    plt.ylabel('Accuracy under attacks.')
    plt.legend()
    plt.axhline(y=0.9921, color='r', linestyle='--')
    plt.axhline(y=0.46, color='k', linestyle='--')
    plt.axvline(x=0.50, color='b', linestyle='--')
    # plt.annotate("Best model", (0.0, 0.50), xycoords='data',
    #              xytext=(0.0, 0.8), arrowprops=dict(arrowstyle='->'))

    plt.subplot(224)
    plt.scatter(distance, deepfool, color='y', alpha=0.3, edgecolors='white', label='Deepfool')
    plt.ylim(0.0, 1.1)
    # plt.plot(distance, fgsm)

    # plt.annotate("Best Performance Point", (df.iloc[0]['distance'], df.iloc[0]['FGSM']), xycoords='data',
    #              xytext=(0.5, 0.6), arrowprops=dict(arrowstyle='->'))

    plt.xlabel("Distance")
    plt.ylabel('Accuracy under attacks.')
    plt.legend()
    plt.axhline(y=0.9921, color='r', linestyle='--')
    plt.axhline(y=0.84, color='k', linestyle='--')
    plt.axvline(x=0.50, color='b', linestyle='--')
    # plt.annotate("Best model", (0.0, 0.84), xycoords='data',
    #              xytext=(0.0, 0.3), arrowprops=dict(arrowstyle='->'))



    # plt.subplot(224)
    # plt.scatter(distance, loss, color='indigo', alpha=0.3, edgecolors='white', label='Loss')
    # plt.ylim(-0.01, 0.10)
    # plt.xlabel("Distance")
    # plt.ylabel("Accuracy Loss")

    plt.tight_layout()
    plt.savefig('gower_distance.png')
    plt.show()


# spearman_corr_fgsm, _ = spearmanr(distance, loss)
# print('Spearmans correlation: %.3f' % spearman_corr_fgsm)
# pearson_corr_fgsm, _ = pearsonr(distance, loss)
# print('Pearsons correlation: %.3f' % pearson_corr_fgsm)
# kendalltau_corr_fgsm, _ = kendalltau(distance, loss)
# print('Kendalltau correlation: %.3f' % kendalltau_corr_fgsm)

print("Covariance correlation: ", np.cov(distance, fgsm))
print("Covariance correlation: ", np.cov(distance, bim_a))
print("Covariance correlation: ", np.cov(distance, bim_b))

# non-linear relationship
spearman_corr_fgsm, _ = spearmanr(distance, fgsm)
spearman_corr_bim_a, _ = spearmanr(distance, bim_a)
spearman_corr_bim_b, _ = spearmanr(distance, bim_b)

print('Spearmans correlation: %.3f' % spearman_corr_fgsm)
print('Spearmans correlation: %.3f' % spearman_corr_bim_a)
print('Spearmans correlation: %.3f' % spearman_corr_bim_b)
print("")

# linear relationship
pearson_corr_fgsm, _ = pearsonr(distance, fgsm)
pearson_corr_bim_a, _ = pearsonr(distance, bim_a)
pearson_corr_bim_b, _ = pearsonr(distance, bim_b)

print('Pearsons correlation: %.3f' % pearson_corr_fgsm)
print('Pearsons correlation: %.3f' % pearson_corr_bim_a)
print('Pearsons correlation: %.3f' % pearson_corr_bim_b)
