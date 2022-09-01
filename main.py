import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.svm import LinearSVC
from tqdm import trange


def plot_decision_boundary(w, b, label=None):
    """
    Plot the decision boundary of a linear classifier.
    """
    xx = np.linspace(-50, 50)
    a = -w[0] / w[1]
    yy = a * xx - b / w[1]
    # margin = 1 / np.sqrt(np.sum(w**2))
    plt.plot(xx, yy, "r-", label=label)
    # plt.plot(xx, yy - np.sqrt(1 + a**2) * margin, "r--", label="norm")
    # plt.plot(xx, yy + np.sqrt(1 + a**2) * margin, "r--", label="norm")


def make_data(scale_factor=1, plot=True):
    """
    Generate 16 data points with 4 support vectors.
    """
    np.random.seed(4)
    x_pos_sv = np.array([[0.5, 1.5], [1.5, 0.5]])
    y_pos_sv = np.ones(x_pos_sv.shape[0])
    x_neg_sv = np.array([[-0.5, -1.5], [-1.5, -0.5]])
    y_neg_sv = np.ones(x_neg_sv.shape[0]) * -1
    x_pos = np.random.uniform(1, 3, (6, 2))
    y_pos = np.ones(x_pos.shape[0])
    x_neg = np.random.uniform(-3, -1, (6, 2))
    y_neg = np.ones(x_neg.shape[0]) * -1
    # x_pos_sv, x_neg_sv, x_pos, x_neg
    x_pos_sv[:, 1] = x_pos_sv[:, 1] * scale_factor
    x_neg_sv[:, 1] = x_neg_sv[:, 1] * scale_factor
    x_pos[:, 1] = x_pos[:, 1] * scale_factor
    x_neg[:, 1] = x_neg[:, 1] * scale_factor
    x = np.concatenate([x_pos, x_neg, x_pos_sv, x_neg_sv])
    y = np.concatenate([y_pos, y_neg, y_pos_sv, y_neg_sv])
    shuffled_idx = np.random.permutation(x.shape[0])
    x = x[shuffled_idx]
    y = y[shuffled_idx]
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(x_pos[:, 0], x_pos[:, 1], c="r", label="+")
        plt.scatter(x_neg[:, 0], x_neg[:, 1], c="b", label="-")
        plt.scatter(x_pos_sv[:, 0], x_pos_sv[:, 1], c="r", edgecolors="k")
        plt.scatter(x_neg_sv[:, 0], x_neg_sv[:, 1], c="b", edgecolors="k")
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.xlabel("x1")
        plt.ylabel("x2")
        # plt.savefig("data.pdf", bbox_inches="tight")
    return x, y


def svm():
    """
    Hard margin SVM solution
    """
    x, y = make_data(scale_factor=1, plot=True)
    model = LinearSVC(penalty="l2", loss="hinge", C=1.0)
    model.fit(x, y)
    w = np.round(model.coef_[0], 4)
    b = np.round(model.intercept_[0], 4)
    print("Weights and bias:", w, b)
    # plot svm decision boundary
    plot_decision_boundary(w, b, "svm")
    plt.title(f"svm: w={w}, b={b}")
    plt.grid()
    plt.savefig("svm_solution.pdf", bbox_inches="tight")
    plt.close()


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # return torch.sigmoid(self.linear(x))
        return self.linear(x)


def logistic_regression(opt="gd"):
    """
    logistic regression solution
    """

    x, y = make_data(scale_factor=1, plot=True)
    _, s, _ = np.linalg.svd(x)
    lr = 1 / (s.max() ** 2)
    x = torch.from_numpy(x).float()
    y = torch.relu(torch.from_numpy(y).float()).reshape(-1, 1)
    model = LogisticRegression(2, 1)
    optimizers = {
        "gd": optim.SGD(model.parameters(), lr=lr),
        "gdmo": optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "adam": optim.Adam(model.parameters(), lr=lr),
    }
    optimizer = optimizers[opt]
    loss_hist = []
    acc_hist = []
    norm_hist = []
    for i in trange(int(1e6)):
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        acc = (y_pred.round() == y).float().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        acc_hist.append(acc.item())
        with torch.no_grad():
            norm_hist.append(torch.norm(model.linear.weight, 2).item())

    w = np.round(model.linear.weight.data.flatten().numpy(), 4)
    b = np.round(model.linear.bias.data.numpy()[0], 4)
    print("Weights and bias:", w, b)
    # plot lr decision boundary
    plot_decision_boundary(w, b, f"lr-{opt}")
    plt.title(f"lr-{opt}: w={w}, b={b:.4f}")
    plt.grid()
    plt.savefig(f"lr_{opt}_solution.pdf", bbox_inches="tight")
    plt.close()
    # plot loss an acc
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(loss_hist)
    ax[0].set_title("loss")
    ax[0].set_xlabel("t")
    ax[0].set_xscale("log")
    ax[1].plot(norm_hist)
    ax[1].set_title("norm")
    ax[1].set_xlabel("t")
    ax[1].set_xscale("log")
    fig.suptitle(f"lr-{opt}: w={w}, b={b:.4f}")
    plt.savefig(f"lr_{opt}_hist.pdf", bbox_inches="tight")
    plt.close()


def main():
    svm()
    for opt in ["gd", "gdmo", "adam"]:
        logistic_regression(opt)


main()
