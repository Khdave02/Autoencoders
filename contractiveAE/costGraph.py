import matplotlib.pyplot as plt


def plot_cost(train_loss, test_loss, title):
    plt.plot(train_loss, color='blue', label='Train loss')
    plt.plot(test_loss, color='red', label='Test loss')
    plt.ylabel('loss')
    plt.xlabel('epochs ')
    plt.title(title)
    plt.legend()
    plt.show()
