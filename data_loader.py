import numpy as np
import pickle
import gzip


def load_data():
    """
    MNIST comprises of three variables ->
    Training Data:     50,000 entries
    Validation Data:   10,000 entries
    Test Data:         10,000 entries

    One Entry of Input = 28 * 28 = 784
    One Entry of Output = 0 - 9 {integer}

    training_data   =  (inputs, outputs)     - Tuple
    inputs.shape    =  (50,000, 784)         - Numpy Array
    outputs.shape   =  (50,000,)             - Numpy Array
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    It Returns Training Data, Validation Data, Test Data

    Training Data is a list of 50,000 entries.
    Each element of this list is a tuple in the form (Input, Output)
    Shape of 1 Input : (784,1)
    Shape of 1 Output : (10,1)

    Validation Data is a list of 10,000 entries.
    Each element of this list is a tuple in the form (Input, Output)
    Shape of 1 Input : (784,1)
    Shape of 1 Output : Integer

    Test Data is a list of 10,000 entries.
    Each element of this list is a tuple in the form (Input, Output)
    Shape of 1 Input : (784,1)
    Shape of 1 Output : Integer
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    Return a 10 bit vectorized result of a number.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
