import numpy as np
import matplotlib.pyplot as plt


def plot_resnet(presnet=False):
    suffix = "_presnet32.txt" if presnet else ".txt"
    title = "PreAct_ResNet_32" if presnet else "ResNet_20"

    ls_train_acc = np.loadtxt("./checkpoint_Cifar/ls_train_acc" + suffix)
    ls_test_acc = np.loadtxt("./checkpoint_Cifar/ls_test_acc" + suffix)
    ls_train_loss = np.loadtxt("./checkpoint_Cifar/ls_train_loss" + suffix)
    ls_test_loss = np.loadtxt("./checkpoint_Cifar/ls_test_loss" + suffix)

    train_acc = np.loadtxt("./checkpoint_Cifar/train_acc" + suffix)
    test_acc = np.loadtxt("./checkpoint_Cifar/test_acc" + suffix)
    train_loss = np.loadtxt("./checkpoint_Cifar/train_loss" + suffix)
    test_loss = np.loadtxt("./checkpoint_Cifar/test_loss" + suffix)

    print(ls_train_acc.shape)
    print(ls_test_acc.shape)
    print(train_acc.shape)
    print(test_acc.shape)

    min_len = min([ls_test_acc.shape[0], test_acc.shape[0]])
    x = np.arange(0, min_len, 1)
    plt.plot(x, train_acc[:min_len], color="red", label="SGD Train")
    plt.plot(x, test_acc[:min_len], color="blue", label="SGD Test")
    plt.plot(x, ls_train_acc[:min_len], color='purple', label="LS-SGD Train")
    plt.plot(x, ls_test_acc[:min_len], color="green", label="LS-SGD Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_resnet(True)
