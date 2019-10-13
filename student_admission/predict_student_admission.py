"""
In this section, we predict student admissions to graduate school at UCLA based on three pieces of data:

GRE Scores (Test)
GPA Scores (Grades)
Class rank (1-4)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO Loading the data

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('student_data.csv')
print(data[:10])


# TODO plotting data

def plot_points(data):
    X = np.array(data[['gre', 'gpa']])
    y = np.array(data[['admit']])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolors='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolors='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
plot_points(data)
plt.show()

# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()


# TODO one-hot encoding the rank
# Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# Drop the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)
print(one_hot_data[:10])

## Alternative solution ##
# if you're using an up-to-date version of pandas,
# you can also use selection by columns

# an equally valid solution
#one_hot_data = pd.get_dummies(data, columns=['rank'])

# TODO Scaling the data

# Copy data
processed_data = one_hot_data[:]
# Scaling the columns

processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0
print(processed_data[:10])

# TODO Splitting the data into Training and Testing

# to test our algorithm, we'll split the data into a Training and a Testing set. The size of the testing set will be 10% of the total data.

sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])

# TODO Splitting the data into features and targets (labels)

features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']
print(features[:10])
print()
print(targets[:10])

# TODO training 2 layer neural network

# Activation (sigmoid) function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1 - sigmoid(x))


def error_formula(y, output):
    return -y * np.log(output) - (1-y) * np.log(1-output)


# TODO Backpropagate the error

'''
Write the error term this is given by the equation
(𝑦−𝑦̂ )𝜎′(𝑥
'''


def error_term_formula(x, y, output):
    return (y-output) * sigmoid_prime(x)


# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5


# Training Function
def train_nn(features, targets, epochs, learnrate):
    # Use to same seed to make debugging easier
    np.random.seed(42)
    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1/n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        # print("del_w ", del_w)

        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here
            #   rather than storing h as a separate variable
            output = sigmoid(np.dot(x, weights))

            # Error = target - output
            error = error_formula(y, output)

            # The error term
            error_term = error_term_formula(x, y, output)

            # Gradient descent step = error_term * gradient * inputs
            del_w += error_term * x
            # Update the weights here. The learning rate times the
            # change in weights, divided by the number of records to average
        weights += learnrate * del_w/n_records

            # Printing out the mean square error on the training set
        if e % (epochs/10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) **2)
            print(loss)
            print("Epoch: ", e)
            if last_loss and last_loss < loss:
                print("Train loss ", loss, "Warning - Loss Increasing")
            else:
                print("Train Loss ", loss)
            last_loss = loss
            print("===========")
    print("finished training ")
    return weights
weights = train_nn(features, targets, epochs, learnrate)

# TODO Calculating the Accuracy on the Test Data

test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == test_out)
print("Prediction Accuracy is : {:.3f}".format(accuracy))
