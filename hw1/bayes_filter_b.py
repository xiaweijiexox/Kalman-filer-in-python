import numpy as np
import matplotlib.pyplot as plt

# colors
green = np.array([0.2980, 0.6, 0])
darkblue = np.array([0, 0.2, 0.4])
VermillionRed = np.array([156, 31, 46]) / 255


def plot_fuction(belief, prediction, posterior_belief):
    """
    plot prior belief, prediction after action, and posterior belief after measurement
    """
    fig = plt.figure()

    # plot prior belief
    ax1 = plt.subplot(311)
    plt.bar(np.arange(0, 10), belief.reshape(-1), color=darkblue)
    plt.title(r'Prior Belief')
    plt.ylim(0, 1)
    plt.ylabel(r'$bel(x_{t-1})$')

    # plot likelihood
    ax2 = plt.subplot(312)
    plt.bar(np.arange(0, 10), prediction.reshape(-1), color=green)
    plt.title(r'Prediction After Action')
    plt.ylim(0, 1)
    plt.ylabel(r'$\overline{bel(x_t})}$')

    # plot posterior belief
    ax3 = plt.subplot(313)
    plt.bar(np.arange(0, 10), posterior_belief.reshape(-1), color=VermillionRed)
    plt.title(r'Posterior Belief After Measurement')
    plt.ylim(0, 1)
    plt.ylabel(r'$bel(x_t})$')

    plt.show()


def bayes_filter_b():
    """
    Follow steps of Bayes filter.  
    You can use the plot_fuction() above to help you check the belief in each step.
    Please print out the final answer.
    """

    # Initialize belief uniformly
    belief = 0.1 * np.ones(10)

    posterior_belief = np.zeros(10)
    #############################################################################
    #                    TODO: Implement you code here                          #
    #############################################################################

    landmarks = {0,3,6}
    p_landmark = np.array([0.8 if i in landmarks else 0.4 for i in range(10)])
    p_nolandmark = 1.0 - p_landmark
    
    prediction = belief.copy() 
    posterior_belief = prediction * p_landmark
    posterior_belief = posterior_belief / posterior_belief.sum()
    belief = posterior_belief

    prediction = np.roll(belief, 3) 
    posterior_belief = prediction * p_landmark
    posterior_belief = posterior_belief / posterior_belief.sum()
    belief = posterior_belief

    prediction = np.roll(belief, 4)
    posterior_belief = prediction * p_nolandmark
    posterior_belief = posterior_belief / posterior_belief.sum()


    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return posterior_belief


if __name__ == '__main__':
    # Test your funtions here
    belief = bayes_filter_b()
    print('Answer for Problem 2b:')
    for i in range(10):
        print("%6d %18.3f\n" % (i, belief[i]))
    plt.bar(np.arange(0, 10), belief)
