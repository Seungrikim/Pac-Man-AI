import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        #print(nn.DotProduct(self.w, x))
        score = nn.DotProduct(self.get_weights(), x)
        return score

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        #print(nn.as_scalar(self.run(x)))
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        while 1:
            convergence = 1
            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    #def update(self, direction, multiplier):
                    nn.Parameter.update(self.get_weights(), x, nn.as_scalar(y))
                    convergence = 0
                else:
                    continue
            if convergence == 1:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #Hidden layer sizes: between 10 and 400.
        #Batch size: between 1 and the size of the dataset. 
        #For Q2 and Q3, we require that total size of the dataset 
        #be evenly divisible by the batch size.
        #Learning rate: between 0.001 and 1.0.
        #Number of hidden layers: between 1 and 3.
        self.batch_size = 1
        self.learning_rate = 0.004
        self.w1 = nn.Parameter(1, 160)
        self.w2 = nn.Parameter(160, 1)
        self.b1 = nn.Parameter(1, 160)
        self.b2 = nn.Parameter(1, 1)
        self.sizeOfParams = 4

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #xm = nn.Linear(x, m)
        #predicted_y = nn.AddBias(xm, b)
        xw1 = nn.Linear(x, self.w1)
        layer = nn.ReLU(nn.AddBias(xw1, self.b1))
        xw2 = nn.Linear(layer, self.w2)
        output = nn.AddBias(xw2, self.b2)
        return output 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        a = self.run(x)
        return nn.SquareLoss(a, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = 0.02
        while loss >= 0.02:
            for x, y in dataset.iterate_once(self.batch_size):
                lossNode = self.get_loss(x, y)
                loss = nn.as_scalar(lossNode)
                #print(loss)
                parameters = [self.w1, self.w2, self.b1, self.b2]
                grad = nn.gradients(lossNode, parameters)
                #print(nn.as_scalar(nn.DotProduct(grad[0],grad[0])))
                for i in range(self.sizeOfParams):
                    parameters[i].update(grad[i], -(self.learning_rate))

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #batch size: 10-500
        #learning rate: 0.05-1
        #hidden size: 100-500
        self.batch_size = 10
        self.learning_rate = 0.05
        self.w1 = nn.Parameter(784, 100)
        self.w2 = nn.Parameter(100, 50)
        self.w3 = nn.Parameter(50, 10)
        self.b1 = nn.Parameter(1, 100)
        self.b2 = nn.Parameter(1, 50)
        self.b3 = nn.Parameter(1, 10)
        self.sizeOfParams = 6

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xw1 = nn.Linear(x, self.w1)
        layer_1 = nn.ReLU(nn.AddBias(xw1, self.b1))
        xw2 = nn.Linear(layer_1, self.w2)
        layer_2 = nn.ReLU(nn.AddBias(xw2, self.b2))
        xw3 = nn.Linear(layer_2, self.w3)
        output = nn.AddBias(xw3, self.b3)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        lossNode = nn.SoftmaxLoss(self.run(x), y)
        return  lossNode

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #loss = 0.02
        while dataset.get_validation_accuracy() < 0.975:
            for x, y in dataset.iterate_once(self.batch_size):
                lossNode = self.get_loss(x, y)
                #loss = nn.as_scalar(lossNode)
                #print(loss)
                parameters = [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]
                grad = nn.gradients(lossNode, parameters)
                #print(nn.as_scalar(nn.DotProduct(grad[0],grad[0])))
                for i in range(self.sizeOfParams):
                    parameters[i].update(grad[i], -(self.learning_rate))


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #learning rate: 0.005-0.1
        #batch size: 4-128
        #hidden layer size: 40-800

        self.batch_size = 40
        self.h_dimensions = 700
        self.learning_rate = 0.005
        self.dimension = len(self.languages)
        self.w_input = nn.Parameter(self.num_chars, self.h_dimensions)
        self.w_hidden = nn.Parameter(self.h_dimensions, self.h_dimensions)
        self.b_hidden = nn.Parameter(1, self.h_dimensions)
        self.w_output = nn.Parameter(self.h_dimensions, self.dimension)
        self.b_output = nn.Parameter(1, self.dimension)
        self.sizeOfParams = 5

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z = nn.Linear(xs[0], self.w_input)
        for character in xs[1:]:
            z = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(character, self.w_input), nn.Linear(z, self.w_hidden)), self.b_hidden))
        output = nn.AddBias(nn.Linear(z, self.w_output), self.b_output)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        lossNode = nn.SoftmaxLoss(self.run(xs), y)
        return  lossNode

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.86:
            for x, y in dataset.iterate_once(self.batch_size):
                lossNode = self.get_loss(x, y)
                parameters = [self.w_input, self.w_hidden, self.b_hidden, self.w_output, self.b_output]
                grad = nn.gradients(lossNode, parameters)
                for i in range(self.sizeOfParams):
                    parameters[i].update(grad[i], -(self.learning_rate))

