import matplotlib.pyplot as plt


def plot_cost(train_loss, test_loss, title):
    '''
        This function plots the loss graph of the test and train losses
        train_loss: train loss per epoch
        test_loss: test loss per epoch
        title: title for the graph
        '''
    plt.plot(train_loss, color='blue', label='Train loss')
    plt.plot(test_loss, color='red', label='Test loss')
    plt.ylabel('loss')
    plt.xlabel('epochs ')
    plt.title(title)
    plt.legend()
    plt.show()
