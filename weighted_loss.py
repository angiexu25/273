import numpy as np

def weighted_loss(total_data, num_of_sample, num_of_classes):
    """
    Input:
        total_data: total number of data
        num_of_sample: a python list of number of emotions in the predefined emotion mapping
        num_of_classes: total number of class

    Output:
        loss: a normalized weighed loss list
    """
    num_of_sample = np.array(num_of_sample)
    loss = total_data / (num_of_classes * num_of_sample)
    loss /= loss.sum()
    return loss