---
title:  "Language Model with Neural Net"
search: false
excerpt: 'GLoVE word embedding, neural net architecture and tSNE representation'
categories: 
  - NLP
  - Neural Net
  - Python
  - Data Visualization
last_modified_at: 2020-06-21T08:06:00-07:00
comments: true
---
> [<i class="fab fa-github"></i>](https://github.com/DaPraxis/blog_material/tree/master/language_model) Code Source 

# Introduction:
This post works with word embeddings and making neural networks learn about words.

We could try to match statistics about the words, or we could train a network that takes a sequence of words as input and learns to predict the word that comes next.

## Starter code and data

Look at the file *raw_sentences.txt*.

It contains the sentences that we will be using for this assignment.
These sentences are fairly simple ones and cover a vocabulary of only 250 words.

First, data import



```ruby
import collections
import pickle
import numpy as np
from tqdm import tqdm
import pylab
use_colab = True
if use_colab:
  from google.colab import drive
import os

TINY = 1e-30
EPS = 1e-4
nax = np.newaxis
```


```ruby
if use_colab:
  drive_name = '/content/drive'
  drive.mount(drive_name)
  drive_folder = # your drive folder
  drive_location = drive_name + '/My Drive/' + drive_folder  # Change this to where your files are located
else:
  # set the drive_location variable to whereever the extracted contents are.
  drive_location = os.getcwd()
  # print(drive_location)

data_location = drive_location + '/' + 'data.pk'
PARTIALLY_TRAINED_MODEL = drive_location + '/' + 'partially_trained.pk'
```

    Go to this URL in a browser: ****
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive


We have already extracted the 4-grams from this dataset and divided them into training, validation, and test sets.
To inspect this data, run the following:


```ruby
data = pickle.load(open(data_location, 'rb'))
print(data['vocab'][0])
print(data['train_inputs'][:10])
print(data['train_targets'][:10])
```

    all
    [[ 27  25  89]
     [183  43 248]
     [182  31  75]
     [116 246 200]
     [222 189 248]
     [ 41  73  25]
     [241  31 222]
     [222  31 157]
     [ 73  31 220]
     [ 41 191  90]]
    [143 116 121 185   5  31  31 143  31  67]


Now *data* is a Python dict which contains the vocabulary, as well as the inputs and targets for all three splits of the data. *data*['vocab'] is a list of the 250 words in the dictionary; *data*['vocab'][0] is the word with index 0, and so on. *data*['train_inputs'] is a 372,500 x 3 matrix where each row gives the indices of the 3 context words for one of the 372,500 training cases.
*data*['train_targets'] is a vector giving the index of the target word for each training case. The validation and test sets are handled analogously.

Even though you only have to modify two specific locations in the code, you may want to read through this code before starting the assignment. 

# Part 1: GLoVE Word Representations

Now, we will implement a simplified version of the GLoVE embedding with the loss defined as

$$L(\{\mathbf{w}_i,b_i\}_{i=1}^V) = \sum_{i,j=1}^V (\mathbf{w}_i^\top\mathbf{w}_j + b_i + b_j - \log X_{ij})^2$$.

Note that each word is represented by a d-dimensional vector $$\mathbf{w}_i$$ and a scalar bias $$b_i$$.

We have provided a few functions for training the embedding:

*   *calculate_log_co_occurence* computes the log co-occurrence matrix of a given corpus
*   *train_GLoVE* runs momentum gradient descent to optimize the embedding
*   *loss_GLoVE:* INPUT - $$V\times d$$ matrix $$W$$ (collection of $$V$$ embedding vectors, each d-dimensional); $$V\times 1$$ vector $$\mathbf{b}$$ (collection of $$V$$ bias terms); $$V \times V$$ log co-occurrence matrix. OUTPUT - loss of the GLoVE objective
*   *grad_GLoVE:* INPUT - $$V\times d$$ matrix $$W$$, $$V\times 1$$ vector b, and $$V\times V$$ log co-occurrence matrix. OUTPUT - $$V\times d$$ matrix grad_W containing the gradient of the loss function w.r.t. $$W$$; $$V\times 1$$ vector grad_b which is the gradient of the loss function w.r.t. $$\mathbf{b}$$. TO BE IMPLEMENTED.

Run the code to compute the co-occurence matrix.
Make sure to add a 1 to the occurences, so there are no 0's in the matrix when we take the elementwise log of the matrix.



```ruby
vocab_size = 250

def calculate_log_co_occurence(word_data):
  "Compute the log-co-occurence matrix for our data."
  log_co_occurence = np.zeros((vocab_size, vocab_size))
  for input in word_data:
    log_co_occurence[input[0], input[1]] += 1
    log_co_occurence[input[1], input[2]] += 1
    # If we want symmetric co-occurence can also increment for these.
    # Optional: How would you generalize the model if our target co-occurence isn't symmetric?
    log_co_occurence[input[1], input[0]] += 1
    log_co_occurence[input[2], input[1]] += 1
  delta_smoothing = 0.5  # A hyperparameter.  You can play with this if you want.
  log_co_occurence += delta_smoothing  # Add delta so log doesn't break on 0's.
  log_co_occurence = np.log(log_co_occurence)
  return log_co_occurence

```


```ruby
log_co_occurence_train = calculate_log_co_occurence(data['train_inputs'])
log_co_occurence_valid = calculate_log_co_occurence(data['valid_inputs'])
```

calculate the gradient of the loss function w.r.t. the parameters $$W$$ and $$\mathbf{b}$$. You should vectorize the computation, i.e. not loop over every word.




```ruby
def loss_GLoVE(W, b, log_co_occurence):
  "Compute the GLoVE loss."
  n,_ = log_co_occurence.shape
  return np.sum((W @ W.T + b @ np.ones([1,n]) + np.ones([n,1])@b.T - log_co_occurence)**2)

def grad_GLoVE(W,  b, log_co_occurence):
  "Return the gradient of GLoVE objective w.r.t W and b."
  "INPUT: W - Vxd; b - Vx1; log_co_occurence: VxV"
  "OUTPUT: grad_W - Vxd; grad_b - Vx1"
  n,_ = log_co_occurence.shape

  grad_W = 4*(W @ W.T + b @ np.ones([1,n]) + np.ones([n,1]) @ b.T - log_co_occurence) @ W
  grad_b = 4*(W @ W.T + b @ np.ones([1,n]) + np.ones([n,1]) @ b.T - log_co_occurence) @ np.ones([n,1])

  return grad_W, grad_b

def train_GLoVE(W, b, log_co_occurence_train, log_co_occurence_valid, n_epochs, do_print=False):
  "Traing W and b according to GLoVE objective."
  n,_ = log_co_occurence_train.shape
  learning_rate = 0.2 / n  # A hyperparameter.  You can play with this if you want.
  for epoch in range(n_epochs):
    grad_W, grad_b = grad_GLoVE(W, b, log_co_occurence_train)
    W -= learning_rate * grad_W
    b -= learning_rate * grad_b
    train_loss, valid_loss = loss_GLoVE(W, b, log_co_occurence_train), loss_GLoVE(W, b, log_co_occurence_valid)
    if do_print:
      print(f"Train Loss: {train_loss}, valid loss: {valid_loss}, grad_norm: {np.sum(grad_w**2)}")
  return W, b, train_loss, valid_loss
```

Train the GLoVE model for a range of embedding dimensions


```ruby
np.random.seed(1)
n_epochs = 500  # A hyperparameter.  You can play with this if you want.
# embedding_dims = np.array([1,2,7,10,11,12,15,50,100, 120, 300])  # Play with this
embedding_dims = np.array([1,2,256])
final_train_losses, final_val_losses = [], []  # Store the final losses for graphing
W_final_2d, b_final_2d = None, None
do_print = False  # If you want to see diagnostic information during training
for embedding_dim in tqdm(embedding_dims):
  init_variance = 0.1  # A hyperparameter.  You can play with this if you want.
  W = init_variance * np.random.normal(size=(250, embedding_dim))
  b = init_variance * np.random.normal(size=(250, 1))
  if do_print:
    print(f"Training for embedding dimension: {embedding_dim}")
  W_final, b_final, train_loss, valid_loss = train_GLoVE(W, b, log_co_occurence_train, log_co_occurence_valid, n_epochs, do_print=do_print)
  if embedding_dim == 2:
    # Save a parameter copy if we are training 2d embedding for visualization later
    W_final_2d = W_final
    b_final_2d = b_final
  final_train_losses += [train_loss]
  final_val_losses += [valid_loss]
  if do_print:
    print(f"Final validation loss: {valid_loss}")

```

    100%|██████████| 3/3 [00:11<00:00,  3.85s/it]


Plot the training and validation losses against the embedding dimension.


```ruby
pylab.loglog(embedding_dims, final_train_losses)
pylab.xlabel("Embedding Dimension")
pylab.ylabel("Training Loss")
pylab.legend()
```

    No handles with labels found to put in legend.
    <matplotlib.legend.Legend at 0x7f022b629208>

![image-center]({{ site.url }}{{ site.baseurl }}../assets/imgs/posts/language_model_files/language_model_15_2.png){: .align-center}

```ruby
pylab.loglog(embedding_dims, final_val_losses)
pylab.xlabel("Embedding Dimension")
pylab.ylabel("Validation Loss")
pylab.legend()
```

    No handles with labels found to put in legend.

    <matplotlib.legend.Legend at 0x7f022b566390>

![image-center]({{ site.url }}{{ site.baseurl }}../assets/imgs/posts/language_model_files/language_model_16_2.png){: .align-center}

Some questions to ask yourself:

1.  Given the vocabulary size $$V$$ and embedding dimensionality $$d$$, how many parameters does the GLoVE model have?

  > $$V\times d + d$$
  
2.  Write the gradient of the loss function with respect to one parameter vector $$\mathbf{w}_i$$.

 > $$\frac{\partial L}{\partial w_i} = 2\sum_{j=1, j\neq i}^{V}(w_i^Tw_j + b_i + b_j - \log{X_{ij}})w_j$$
$$= 4\sum_{j=1}^{V}(w_i^Tw_j + b_i + b_j - \log{X_{ij}})w_j$$

  > $$\frac{\partial L}{\partial b_i} = 2\sum_{j=1, j\neq i}^{V}w_i^Tw_j + b_i + b_j - \log{X_{ij}}$$
$$= 4\sum_{j=1}^{V}w_i^Tw_j + b_i + b_j - \log{X_{ij}}$$

3.  Train the model with varying dimensionality $$d$$.
Which $$d$$ leads to optimal validation performance?
Why does / doesn't larger $$d$$ always lead to better validation error?

  > when d = 11, d leads to optimal validation performance with the smallest validation error. This dimentionality indicates a possible encoding of $$2^{11} = 2048$$ ways. Model may be over fit if dimension is too large, while underfit if dimension is too small.  


# Part 2: Network Architecture
1. - Word embedding weight: $$250\times 16$$

    - Embedded to hidden weight: $$3\times16\times 128$$

    - Hidden to output weight: $$128\times 250$$

    - Hidden bias: $$128\times 1$$

    - Output bias: $$250\times1$$

    - Total: 42522 parameters. The hidden to output weight has the most parameters

2. A 4-grams model is a model predict the 4th word from 3 previous words. That is, we have 250 vocabularies in total, there is a combination of $$250^4 = 3906250000$$ possible non-repeated outcomes. 

# Part 3: Training the model

There are three classes defined in this *part*: *Params*, *Activations*, *Model*.


```ruby
class Params(object):
    """A class representing the trainable parameters of the model. This class has five fields:
    
           word_embedding_weights, a matrix of size N_V x D, where N_V is the number of words in the vocabulary
                   and D is the embedding dimension.
           embed_to_hid_weights, a matrix of size N_H x 3D, where N_H is the number of hidden units. The first D
                   columns represent connections from the embedding of the first context word, the next D columns
                   for the second context word, and so on.
           hid_bias, a vector of length N_H
           hid_to_output_weights, a matrix of size N_V x N_H
           output_bias, a vector of length N_V"""

    def __init__(self, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights,
                 hid_bias, output_bias):
        self.word_embedding_weights = word_embedding_weights
        self.embed_to_hid_weights = embed_to_hid_weights
        self.hid_to_output_weights = hid_to_output_weights
        self.hid_bias = hid_bias
        self.output_bias = output_bias

    def copy(self):
        return self.__class__(self.word_embedding_weights.copy(), self.embed_to_hid_weights.copy(),
                              self.hid_to_output_weights.copy(), self.hid_bias.copy(), self.output_bias.copy())

    @classmethod
    def zeros(cls, vocab_size, context_len, embedding_dim, num_hid):
        """A constructor which initializes all weights and biases to 0."""
        word_embedding_weights = np.zeros((vocab_size, embedding_dim))
        embed_to_hid_weights = np.zeros((num_hid, context_len * embedding_dim))
        hid_to_output_weights = np.zeros((vocab_size, num_hid))
        hid_bias = np.zeros(num_hid)
        output_bias = np.zeros(vocab_size)
        return cls(word_embedding_weights, embed_to_hid_weights, hid_to_output_weights,
                   hid_bias, output_bias)

    @classmethod
    def random_init(cls, init_wt, vocab_size, context_len, embedding_dim, num_hid):
        """A constructor which initializes weights to small random values and biases to 0."""
        word_embedding_weights = np.random.normal(0., init_wt, size=(vocab_size, embedding_dim))
        embed_to_hid_weights = np.random.normal(0., init_wt, size=(num_hid, context_len * embedding_dim))
        hid_to_output_weights = np.random.normal(0., init_wt, size=(vocab_size, num_hid))
        hid_bias = np.zeros(num_hid)
        output_bias = np.zeros(vocab_size)
        return cls(word_embedding_weights, embed_to_hid_weights, hid_to_output_weights,
                   hid_bias, output_bias)

    ###### The functions below are Python's somewhat oddball way of overloading operators, so that
    ###### we can do arithmetic on Params instances. You don't need to understand this to do the assignment.

    def __mul__(self, a):
        return self.__class__(a * self.word_embedding_weights,
                              a * self.embed_to_hid_weights,
                              a * self.hid_to_output_weights,
                              a * self.hid_bias,
                              a * self.output_bias)

    def __rmul__(self, a):
        return self * a

    def __add__(self, other):
        return self.__class__(self.word_embedding_weights + other.word_embedding_weights,
                              self.embed_to_hid_weights + other.embed_to_hid_weights,
                              self.hid_to_output_weights + other.hid_to_output_weights,
                              self.hid_bias + other.hid_bias,
                              self.output_bias + other.output_bias)

    def __sub__(self, other):
        return self + -1. * other
```


```ruby
class Activations(object):
    """A class representing the activations of the units in the network. This class has three fields:

        embedding_layer, a matrix of B x 3D matrix (where B is the batch size and D is the embedding dimension),
                representing the activations for the embedding layer on all the cases in a batch. The first D
                columns represent the embeddings for the first context word, and so on.
        hidden_layer, a B x N_H matrix representing the hidden layer activations for a batch
        output_layer, a B x N_V matrix representing the output layer activations for a batch"""

    def __init__(self, embedding_layer, hidden_layer, output_layer):
        self.embedding_layer = embedding_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

def get_batches(inputs, targets, batch_size, shuffle=True):
    """Divide a dataset (usually the training set) into mini-batches of a given size. This is a
    'generator', i.e. something you can use in a for loop. You don't need to understand how it
    works to do the assignment."""

    if inputs.shape[0] % batch_size != 0:
        raise RuntimeError('The number of data points must be a multiple of the batch size.')
    num_batches = inputs.shape[0] // batch_size

    if shuffle:
        idxs = np.random.permutation(inputs.shape[0])
        inputs = inputs[idxs, :]
        targets = targets[idxs]

    for m in range(num_batches):
        yield inputs[m * batch_size:(m + 1) * batch_size, :], \
              targets[m * batch_size:(m + 1) * batch_size]
```

Now, we will implement a method which computes the gradient using backpropagation.
To start out, the *Model* class contains several important methods used in training:


*   *compute_activations* computes the activations of all units on a given input batch
*   *compute_loss* computes the total cross-entropy loss on a mini-batch
*   *evaluate* computes the average cross-entropy loss for a given set of inputs and targets

You will need to complete the implementation of two additional methods which are needed for training:


*   *compute_loss_derivative* computes the derivative of the loss function with respect to the output layer inputs.
*   *back_propagate* is the function which computes the gradient of the loss with respect to model parameters using backpropagation.
It uses the derivatives computed by *compute_loss_derivative*.
Some parts are already filled in for you, but you need to compute the matrices of derivatives for *embed_to_hid_weights*, *hid_bias*, *hid_to_output_weights*, and *output_bias*.
These matrices have the same sizes as the parameter matrices (see previous section).

In order to implement backpropagation efficiently, you need to express the computations in terms of matrix operations, rather than *for* loops.
You should first work through the derivatives on pencil and paper.
First, apply the chain rule to compute the derivatives with respect to individual units, weights, and biases.
Next, take the formulas you've derived, and express them in matrix form.
You should be able to express all of the required computations using only matrix multiplication, matrix transpose, and elementwise operations --- no *for* loops!

If you want inspiration, read through the code for *Model.compute_activations* and try to understand how the matrix operations correspond to the computations performed by all the units in the network.
        
To make your life easier, we have provided the routine *checking.check_gradients*, which checks your gradients using finite differences.
You should make sure this check passes before continuing with the assignment.






```ruby
  class Model(object):
    """A class representing the language model itself. This class contains various methods used in training
    the model and visualizing the learned representations. It has two fields:

        params, a Params instance which contains the model parameters
        vocab, a list containing all the words in the dictionary; vocab[0] is the word with index
               0, and so on."""

    def __init__(self, params, vocab):
        self.params = params
        self.vocab = vocab

        self.vocab_size = len(vocab)
        self.embedding_dim = self.params.word_embedding_weights.shape[1]
        self.embedding_layer_dim = self.params.embed_to_hid_weights.shape[1]
        self.context_len = self.embedding_layer_dim // self.embedding_dim
        self.num_hid = self.params.embed_to_hid_weights.shape[0]

    def copy(self):
        return self.__class__(self.params.copy(), self.vocab[:])

    @classmethod
    def random_init(cls, init_wt, vocab, context_len, embedding_dim, num_hid):
        """Constructor which randomly initializes the weights to Gaussians with standard deviation init_wt
        and initializes the biases to all zeros."""
        params = Params.random_init(init_wt, len(vocab), context_len, embedding_dim, num_hid)
        return Model(params, vocab)

    def indicator_matrix(self, targets):
        """Construct a matrix where the kth entry of row i is 1 if the target for example
        i is k, and all other entries are 0."""
        batch_size = targets.size
        expanded_targets = np.zeros((batch_size, len(self.vocab)))
        expanded_targets[np.arange(batch_size), targets] = 1.
        return expanded_targets

    def compute_loss_derivative(self, output_activations, expanded_target_batch):
        """Compute the derivative of the cross-entropy loss function with respect to the inputs
        to the output units. In particular, the output layer computes the softmax

            y_i = e^{z_i} / \sum_j e^{z_j}.

        This function should return a batch_size x vocab_size matrix, where the (i, j) entry
        is dC / dz_j computed for the ith training case, where C is the loss function

            C = -sum(t_i log y_i).

        The arguments are as follows:

            output_activations - the activations of the output layer, i.e. the y_i's.
            expanded_target_batch - each row is the indicator vector for a target word,
                i.e. the (i, j) entry is 1 if the i'th word is j, and 0 otherwise."""

        return output_activations-expanded_target_batch


    def compute_loss(self, output_activations, expanded_target_batch):
        """Compute the total loss over a mini-batch. expanded_target_batch is the matrix obtained
        by calling indicator_matrix on the targets for the batch."""
        return -np.sum(expanded_target_batch * np.log(output_activations + TINY))

    def compute_activations(self, inputs):
        """Compute the activations on a batch given the inputs. Returns an Activations instance.
        You should try to read and understand this function, since this will give you clues for
        how to implement back_propagate."""

        batch_size = inputs.shape[0]
        if inputs.shape[1] != self.context_len:
            raise RuntimeError('Dimension of the input vectors should be {}, but is instead {}'.format(
                self.context_len, inputs.shape[1]))

        # Embedding layer
        # Look up the input word indies in the word_embedding_weights matrix
        embedding_layer_state = np.zeros((batch_size, self.embedding_layer_dim))
        for i in range(self.context_len):
            embedding_layer_state[:, i * self.embedding_dim:(i + 1) * self.embedding_dim] = \
                self.params.word_embedding_weights[inputs[:, i], :]

        # Hidden layer
        inputs_to_hid = np.dot(embedding_layer_state, self.params.embed_to_hid_weights.T) + \
                        self.params.hid_bias
        # Apply logistic activation function
        hidden_layer_state = 1. / (1. + np.exp(-inputs_to_hid))

        # Output layer
        inputs_to_softmax = np.dot(hidden_layer_state, self.params.hid_to_output_weights.T) + \
                            self.params.output_bias

        # Subtract maximum.
        # Remember that adding or subtracting the same constant from each input to a
        # softmax unit does not affect the outputs. So subtract the maximum to
        # make all inputs <= 0. This prevents overflows when computing their exponents.
        inputs_to_softmax -= inputs_to_softmax.max(1).reshape((-1, 1))

        output_layer_state = np.exp(inputs_to_softmax)
        output_layer_state /= output_layer_state.sum(1).reshape((-1, 1))

        return Activations(embedding_layer_state, hidden_layer_state, output_layer_state)

    def back_propagate(self, input_batch, activations, loss_derivative):
        """Compute the gradient of the loss function with respect to the trainable parameters
        of the model. The arguments are as follows:

             input_batch - the indices of the context words
             activations - an Activations class representing the output of Model.compute_activations
             loss_derivative - the matrix of derivatives computed by compute_loss_derivative
             
        Part of this function is already completed, but you need to fill in the derivative
        computations for hid_to_output_weights_grad, output_bias_grad, embed_to_hid_weights_grad,
        and hid_bias_grad. See the documentation for the Params class for a description of what
        these matrices represent."""

        # The matrix with values dC / dz_j, where dz_j is the input to the jth hidden unit,
        # i.e. y_j = 1 / (1 + e^{-z_j})
        hid_deriv = np.dot(loss_derivative, self.params.hid_to_output_weights) \
                    * activations.hidden_layer * (1. - activations.hidden_layer)

        hid_to_output_weights_grad = np.dot(loss_derivative.T, activations.hidden_layer)
        output_bias_grad = np.dot(loss_derivative.T, np.ones((activations.hidden_layer.shape[0],1))).ravel()
        embed_to_hid_weights_grad = np.dot(hid_deriv.T, activations.embedding_layer)
        hid_bias_grad = np.dot(hid_deriv.T, np.ones((activations.hidden_layer.shape[0], 1))).ravel()

        # The matrix of derivatives for the embedding layer
        embed_deriv = np.dot(hid_deriv, self.params.embed_to_hid_weights)

        # Embedding layer
        word_embedding_weights_grad = np.zeros((self.vocab_size, self.embedding_dim))
        for w in range(self.context_len):
            word_embedding_weights_grad += np.dot(self.indicator_matrix(input_batch[:, w]).T,
                                                  embed_deriv[:, w * self.embedding_dim:(w + 1) * self.embedding_dim])

        return Params(word_embedding_weights_grad, embed_to_hid_weights_grad, hid_to_output_weights_grad,
                      hid_bias_grad, output_bias_grad)

    def evaluate(self, inputs, targets, batch_size=100):
        """Compute the average cross-entropy over a dataset.

            inputs: matrix of shape D x N
            targets: one-dimensional matrix of length N"""

        ndata = inputs.shape[0]

        total = 0.
        for input_batch, target_batch in get_batches(inputs, targets, batch_size):
            activations = self.compute_activations(input_batch)
            expanded_target_batch = self.indicator_matrix(target_batch)
            cross_entropy = -np.sum(expanded_target_batch * np.log(activations.output_layer + TINY))
            total += cross_entropy

        return total / float(ndata)

    def display_nearest_words(self, word, k=10):
        """List the k words nearest to a given word, along with their distances."""

        if word not in self.vocab:
            print('Word "{}" not in vocabulary.'.format(word))
            return

        # Compute distance to every other word.
        idx = self.vocab.index(word)
        word_rep = self.params.word_embedding_weights[idx, :]
        diff = self.params.word_embedding_weights - word_rep.reshape((1, -1))
        distance = np.sqrt(np.sum(diff ** 2, axis=1))

        # Sort by distance.
        order = np.argsort(distance)
        order = order[1:1 + k]  # The nearest word is the query word itself, skip that.
        for i in order:
            print('{}: {}'.format(self.vocab[i], distance[i]))

    def predict_next_word(self, word1, word2, word3, k=10):
        """List the top k predictions for the next word along with their probabilities.
        Inputs:
            word1: The first word as a string.
            word2: The second word as a string.
            word3: The third word as a string.
            k: The k most probable predictions are shown.
        Example usage:
            model.predict_next_word('john', 'might', 'be', 3)
            model.predict_next_word('life', 'in', 'new', 3)"""

        if word1 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word1))
        if word2 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word2))
        if word3 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word3))

        idx1, idx2, idx3 = self.vocab.index(word1), self.vocab.index(word2), self.vocab.index(word3)
        input = np.array([idx1, idx2, idx3]).reshape((1, -1))
        activations = self.compute_activations(input)
        prob = activations.output_layer.ravel()
        idxs = np.argsort(prob)[::-1]  # sort descending
        for i in idxs[:k]:
            print('{} {} {} {} Prob: {:1.5f}'.format(word1, word2, word3, self.vocab[i], prob[i]))

    def word_distance(self, word1, word2):
        """Compute the distance between the vector representations of two words."""

        if word1 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word1))
        if word2 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word2))

        idx1, idx2 = self.vocab.index(word1), self.vocab.index(word2)
        word_rep1 = self.params.word_embedding_weights[idx1, :]
        word_rep2 = self.params.word_embedding_weights[idx2, :]
        diff = word_rep1 - word_rep2
        return np.sqrt(np.sum(diff ** 2))
```

Perform routine *checking.check_gradients*, which checks your gradients using finite differences.
You should make sure this check passes before continuing


```ruby
def relative_error(a, b):
    return np.abs(a - b) / (np.abs(a) + np.abs(b))


def check_output_derivatives(model, input_batch, target_batch):
    def softmax(z):
        z = z.copy()
        z -= z.max(1).reshape((-1, 1))
        y = np.exp(z)
        y /= y.sum(1).reshape((-1, 1))
        return y

    batch_size = input_batch.shape[0]
    z = np.random.normal(size=(batch_size, model.vocab_size))
    y = softmax(z)

    expanded_target_batch = model.indicator_matrix(target_batch)
    loss_derivative = model.compute_loss_derivative(y, expanded_target_batch)

    if loss_derivative is None:
        print('Loss derivative not implemented yet.')
        return False

    if loss_derivative.shape != (batch_size, model.vocab_size):
        print('Loss derivative should be size {} but is actually {}.'.format(
            (batch_size, model.vocab_size), loss_derivative.shape))
        return False

    def obj(z):
        y = softmax(z)
        return model.compute_loss(y, expanded_target_batch)

    for count in range(1000):
        i, j = np.random.randint(0, loss_derivative.shape[0]), np.random.randint(0, loss_derivative.shape[1])

        z_plus = z.copy()
        z_plus[i, j] += EPS
        obj_plus = obj(z_plus)

        z_minus = z.copy()
        z_minus[i, j] -= EPS
        obj_minus = obj(z_minus)

        empirical = (obj_plus - obj_minus) / (2. * EPS)
        rel = relative_error(empirical, loss_derivative[i, j])
        if rel > 1e-4:
            print('The loss derivative has a relative error of {}, which is too large.'.format(rel))
            return False

    print('The loss derivative looks OK.')
    return True


def check_param_gradient(model, param_name, input_batch, target_batch):
    activations = model.compute_activations(input_batch)
    expanded_target_batch = model.indicator_matrix(target_batch)
    loss_derivative = model.compute_loss_derivative(activations.output_layer, expanded_target_batch)
    param_gradient = model.back_propagate(input_batch, activations, loss_derivative)

    def obj(model):
        activations = model.compute_activations(input_batch)
        return model.compute_loss(activations.output_layer, expanded_target_batch)

    dims = getattr(model.params, param_name).shape
    is_matrix = (len(dims) == 2)

    if getattr(param_gradient, param_name).shape != dims:
        print('The gradient for {} should be size {} but is actually {}.'.format(
            param_name, dims, getattr(param_gradient, param_name).shape))
        return

    for count in range(1000):
        if is_matrix:
            slc = np.random.randint(0, dims[0]), np.random.randint(0, dims[1])
        else:
            slc = np.random.randint(dims[0])

        model_plus = model.copy()
        getattr(model_plus.params, param_name)[slc] += EPS
        obj_plus = obj(model_plus)

        model_minus = model.copy()
        getattr(model_minus.params, param_name)[slc] -= EPS
        obj_minus = obj(model_minus)

        empirical = (obj_plus - obj_minus) / (2. * EPS)
        exact = getattr(param_gradient, param_name)[slc]
        rel = relative_error(empirical, exact)
        if rel > 1e-4:
            print('The loss derivative has a relative error of {}, which is too large.'.format(rel))
            return False

    print('The gradient for {} looks OK.'.format(param_name))


def load_partially_trained_model():
    obj = pickle.load(open(PARTIALLY_TRAINED_MODEL, 'rb'))
    params = Params(obj['word_embedding_weights'], obj['embed_to_hid_weights'],
                                   obj['hid_to_output_weights'], obj['hid_bias'],
                                   obj['output_bias'])
    vocab = obj['vocab']
    return Model(params, vocab)


def check_gradients():
    """Check the computed gradients using finite differences."""
    np.random.seed(0)

    np.seterr(all='ignore')  # suppress a warning which is harmless

    model = load_partially_trained_model()
    data_obj = pickle.load(open(data_location, 'rb'))
    train_inputs, train_targets = data_obj['train_inputs'], data_obj['train_targets']
    input_batch = train_inputs[:100, :]
    target_batch = train_targets[:100]

    if not check_output_derivatives(model, input_batch, target_batch):
        return

    for param_name in ['word_embedding_weights', 'embed_to_hid_weights', 'hid_to_output_weights',
                       'hid_bias', 'output_bias']:
        check_param_gradient(model, param_name, input_batch, target_batch)


def print_gradients():
    """Print out certain derivatives for grading."""

    model = load_partially_trained_model()
    data_obj = pickle.load(open(data_location, 'rb'))
    train_inputs, train_targets = data_obj['train_inputs'], data_obj['train_targets']
    input_batch = train_inputs[:100, :]
    target_batch = train_targets[:100]

    activations = model.compute_activations(input_batch)
    expanded_target_batch = model.indicator_matrix(target_batch)
    loss_derivative = model.compute_loss_derivative(activations.output_layer, expanded_target_batch)
    param_gradient = model.back_propagate(input_batch, activations, loss_derivative)

    print('loss_derivative[2, 5]', loss_derivative[2, 5])
    print('loss_derivative[2, 121]', loss_derivative[2, 121])
    print('loss_derivative[5, 33]', loss_derivative[5, 33])
    print('loss_derivative[5, 31]', loss_derivative[5, 31])
    print()
    print('param_gradient.word_embedding_weights[27, 2]', param_gradient.word_embedding_weights[27, 2])
    print('param_gradient.word_embedding_weights[43, 3]', param_gradient.word_embedding_weights[43, 3])
    print('param_gradient.word_embedding_weights[22, 4]', param_gradient.word_embedding_weights[22, 4])
    print('param_gradient.word_embedding_weights[2, 5]', param_gradient.word_embedding_weights[2, 5])
    print()
    print('param_gradient.embed_to_hid_weights[10, 2]', param_gradient.embed_to_hid_weights[10, 2])
    print('param_gradient.embed_to_hid_weights[15, 3]', param_gradient.embed_to_hid_weights[15, 3])
    print('param_gradient.embed_to_hid_weights[30, 9]', param_gradient.embed_to_hid_weights[30, 9])
    print('param_gradient.embed_to_hid_weights[35, 21]', param_gradient.embed_to_hid_weights[35, 21])
    print()
    print('param_gradient.hid_bias[10]', param_gradient.hid_bias[10])
    print('param_gradient.hid_bias[20]', param_gradient.hid_bias[20])
    print()
    print('param_gradient.output_bias[0]', param_gradient.output_bias[0])
    print('param_gradient.output_bias[1]', param_gradient.output_bias[1])
    print('param_gradient.output_bias[2]', param_gradient.output_bias[2])
    print('param_gradient.output_bias[3]', param_gradient.output_bias[3])

```


```ruby
# check_gradients()
print_gradients()
```

    loss_derivative[2, 5] 0.001112231773782498
    loss_derivative[2, 121] -0.9991004720395987
    loss_derivative[5, 33] 0.0001903237803173703
    loss_derivative[5, 31] -0.7999757709589483
    
    param_gradient.word_embedding_weights[27, 2] -0.27199539981936866
    param_gradient.word_embedding_weights[43, 3] 0.8641722267354154
    param_gradient.word_embedding_weights[22, 4] -0.2546730202374648
    param_gradient.word_embedding_weights[2, 5] 0.0
    
    param_gradient.embed_to_hid_weights[10, 2] -0.6526990313918257
    param_gradient.embed_to_hid_weights[15, 3] -0.13106433000472612
    param_gradient.embed_to_hid_weights[30, 9] 0.11846774618169399
    param_gradient.embed_to_hid_weights[35, 21] -0.10004526104604386
    
    param_gradient.hid_bias[10] 0.25376638738156426
    param_gradient.hid_bias[20] -0.03326739163635369
    
    param_gradient.output_bias[0] -2.062759603217304
    param_gradient.output_bias[1] 0.03902008573921689
    param_gradient.output_bias[2] -0.7561537928318482
    param_gradient.output_bias[3] 0.21235172051123635


Now we fininshed the implementation and down to the training
The function *train* implements the main training procedure.
It takes two arguments:


*   *embedding_dim*: The number of dimensions in the distributed representation.
*   *num_hid*: The number of hidden units


As the model trains, the script prints out some numbers that tell you how well the training is going.
It shows:


*   The cross entropy on the last 100 mini-batches of the training set. This is shown after every 100 mini-batches.
*   The cross entropy on the entire validation set every 1000 mini-batches of training.

At the end of training, this function shows the cross entropies on the training, validation and test sets.
It will return a *Model* instance.


```ruby
_train_inputs = None
_train_targets = None
_vocab = None

DEFAULT_TRAINING_CONFIG = {'batch_size': 100,  # the size of a mini-batch
                           'learning_rate': 0.1,  # the learning rate
                           'momentum': 0.9,  # the decay parameter for the momentum vector
                           'epochs': 50,  # the maximum number of epochs to run
                           'init_wt': 0.01,  # the standard deviation of the initial random weights
                           'context_len': 3,  # the number of context words used
                           'show_training_CE_after': 100,  # measure training error after this many mini-batches
                           'show_validation_CE_after': 1000,  # measure validation error after this many mini-batches
                           }


def find_occurrences(word1, word2, word3):
    """Lists all the words that followed a given tri-gram in the training set and the number of
    times each one followed it."""

    # cache the data so we don't keep reloading
    global _train_inputs, _train_targets, _vocab
    if _train_inputs is None:
        data_obj = pickle.load(open(data_location, 'rb'))
        _vocab = data_obj['vocab']
        _train_inputs, _train_targets = data_obj['train_inputs'], data_obj['train_targets']

    if word1 not in _vocab:
        raise RuntimeError('Word "{}" not in vocabulary.'.format(word1))
    if word2 not in _vocab:
        raise RuntimeError('Word "{}" not in vocabulary.'.format(word2))
    if word3 not in _vocab:
        raise RuntimeError('Word "{}" not in vocabulary.'.format(word3))

    idx1, idx2, idx3 = _vocab.index(word1), _vocab.index(word2), _vocab.index(word3)
    idxs = np.array([idx1, idx2, idx3])

    matches = np.all(_train_inputs == idxs.reshape((1, -1)), 1)

    if np.any(matches):
        counts = collections.defaultdict(int)
        for m in np.where(matches)[0]:
            counts[_vocab[_train_targets[m]]] += 1

        word_counts = sorted(list(counts.items()), key=lambda t: t[1], reverse=True)
        print('The tri-gram "{} {} {}" was followed by the following words in the training set:'.format(
            word1, word2, word3))
        for word, count in word_counts:
            if count > 1:
                print('    {} ({} times)'.format(word, count))
            else:
                print('    {} (1 time)'.format(word))
    else:
        print('The tri-gram "{} {} {}" did not occur in the training set.'.format(word1, word2, word3))


def train(embedding_dim, num_hid, config=DEFAULT_TRAINING_CONFIG):
    """This is the main training routine for the language model. It takes two parameters:

        embedding_dim, the dimension of the embeddilanguage_model.pyng space
        num_hid, the number of hidden units."""

    # Load the data
    data_obj = pickle.load(open(data_location, 'rb'))
    vocab = data_obj['vocab']
    train_inputs, train_targets = data_obj['train_inputs'], data_obj['train_targets']
    valid_inputs, valid_targets = data_obj['valid_inputs'], data_obj['valid_targets']
    test_inputs, test_targets = data_obj['test_inputs'], data_obj['test_targets']

    # Randomly initialize the trainable parameters
    model = Model.random_init(config['init_wt'], vocab, config['context_len'], embedding_dim, num_hid)

    # Variables used for early stopping
    best_valid_CE = np.infty
    end_training = False

    # Initialize the momentum vector to all zeros
    delta = Params.zeros(len(vocab), config['context_len'], embedding_dim, num_hid)

    this_chunk_CE = 0.
    batch_count = 0
    for epoch in range(1, config['epochs'] + 1):
        if end_training:
            break

        print()
        print('Epoch', epoch)

        for m, (input_batch, target_batch) in enumerate(get_batches(train_inputs, train_targets,
                                                                    config['batch_size'])):
            batch_count += 1

            # Forward propagate
            activations = model.compute_activations(input_batch)

            # Compute loss derivative
            expanded_target_batch = model.indicator_matrix(target_batch)
            loss_derivative = model.compute_loss_derivative(activations.output_layer, expanded_target_batch)
            loss_derivative /= config['batch_size']

            # Measure loss function
            cross_entropy = model.compute_loss(activations.output_layer, expanded_target_batch) / config['batch_size']
            this_chunk_CE += cross_entropy
            if batch_count % config['show_training_CE_after'] == 0:
                print('Batch {} Train CE {:1.3f}'.format(
                    batch_count, this_chunk_CE / config['show_training_CE_after']))
                this_chunk_CE = 0.

            # Backpropagate
            loss_gradient = model.back_propagate(input_batch, activations, loss_derivative)

            # Update the momentum vector and model parameters
            delta = config['momentum'] * delta + loss_gradient
            model.params -= config['learning_rate'] * delta

            # Validate
            if batch_count % config['show_validation_CE_after'] == 0:
                print('Running validation...')
                cross_entropy = model.evaluate(valid_inputs, valid_targets)
                print('Validation cross-entropy: {:1.3f}'.format(cross_entropy))

                if cross_entropy > best_valid_CE:
                    print('Validation error increasing!  Training stopped.')
                    end_training = True
                    break

                best_valid_CE = cross_entropy

    print()
    train_CE = model.evaluate(train_inputs, train_targets)
    print('Final training cross-entropy: {:1.3f}'.format(train_CE))
    valid_CE = model.evaluate(valid_inputs, valid_targets)
    print('Final validation cross-entropy: {:1.3f}'.format(valid_CE))
    test_CE = model.evaluate(test_inputs, test_targets)
    print('Final test cross-entropy: {:1.3f}'.format(test_CE))

    return model
```

Run the training.



```ruby
embedding_dim = 16
num_hid = 128
trained_model = train(embedding_dim, num_hid)
```

    
    Epoch 1
    Batch 100 Train CE 4.543
    Batch 200 Train CE 4.430
    Batch 300 Train CE 4.456
    Batch 400 Train CE 4.460
    Batch 500 Train CE 4.446
    Batch 600 Train CE 4.423
    Batch 700 Train CE 4.445
    Batch 800 Train CE 4.440
    Batch 900 Train CE 4.415
    Batch 1000 Train CE 4.392
    Running validation...
    Validation cross-entropy: 4.435
    Batch 1100 Train CE 4.385
    Batch 1200 Train CE 4.342
    Batch 1300 Train CE 4.276
    Batch 1400 Train CE 4.199
    Batch 1500 Train CE 4.144
    Batch 1600 Train CE 4.048
    Batch 1700 Train CE 4.088
    Batch 1800 Train CE 4.065
    Batch 1900 Train CE 4.014
    Batch 2000 Train CE 3.993
    Running validation...
    Validation cross-entropy: 4.008
    Batch 2100 Train CE 3.994
    Batch 2200 Train CE 3.940
    Batch 2300 Train CE 3.913
    Batch 2400 Train CE 3.849
    Batch 2500 Train CE 3.812
    Batch 2600 Train CE 3.742
    Batch 2700 Train CE 3.706
    Batch 2800 Train CE 3.699
    Batch 2900 Train CE 3.620
    Batch 3000 Train CE 3.594
    Running validation...
    Validation cross-entropy: 3.574
    Batch 3100 Train CE 3.541
    Batch 3200 Train CE 3.486
    Batch 3300 Train CE 3.490
    Batch 3400 Train CE 3.431
    Batch 3500 Train CE 3.431
    Batch 3600 Train CE 3.397
    Batch 3700 Train CE 3.388
    
    Epoch 2
    Batch 3800 Train CE 3.370
    Batch 3900 Train CE 3.338
    Batch 4000 Train CE 3.330
    Running validation...
    Validation cross-entropy: 3.326
    Batch 4100 Train CE 3.344
    Batch 4200 Train CE 3.300
    Batch 4300 Train CE 3.274
    Batch 4400 Train CE 3.281
    Batch 4500 Train CE 3.254
    Batch 4600 Train CE 3.257
    Batch 4700 Train CE 3.257
    Batch 4800 Train CE 3.230
    Batch 4900 Train CE 3.229
    Batch 5000 Train CE 3.218
    Running validation...
    Validation cross-entropy: 3.183
    Batch 5100 Train CE 3.188
    Batch 5200 Train CE 3.183
    Batch 5300 Train CE 3.150
    Batch 5400 Train CE 3.127
    Batch 5500 Train CE 3.183
    Batch 5600 Train CE 3.104
    Batch 5700 Train CE 3.155
    Batch 5800 Train CE 3.118
    Batch 5900 Train CE 3.081
    Batch 6000 Train CE 3.105
    Running validation...
    Validation cross-entropy: 3.113
    Batch 6100 Train CE 3.102
    Batch 6200 Train CE 3.091
    Batch 6300 Train CE 3.092
    Batch 6400 Train CE 3.049
    Batch 6500 Train CE 3.044
    Batch 6600 Train CE 3.061
    Batch 6700 Train CE 3.032
    Batch 6800 Train CE 3.004
    Batch 6900 Train CE 3.022
    Batch 7000 Train CE 3.018
    Running validation...
    Validation cross-entropy: 3.031
    Batch 7100 Train CE 3.042
    Batch 7200 Train CE 3.036
    Batch 7300 Train CE 3.027
    Batch 7400 Train CE 3.016
    
    Epoch 3
    Batch 7500 Train CE 3.001
    Batch 7600 Train CE 3.016
    Batch 7700 Train CE 2.982
    Batch 7800 Train CE 2.984
    Batch 7900 Train CE 2.962
    Batch 8000 Train CE 2.974
    Running validation...
    Validation cross-entropy: 2.992
    Batch 8100 Train CE 2.990
    Batch 8200 Train CE 2.971
    Batch 8300 Train CE 2.964
    Batch 8400 Train CE 2.939
    Batch 8500 Train CE 2.947
    Batch 8600 Train CE 2.960
    Batch 8700 Train CE 2.957
    Batch 8800 Train CE 2.935
    Batch 8900 Train CE 2.941
    Batch 9000 Train CE 2.929
    Running validation...
    Validation cross-entropy: 2.951
    Batch 9100 Train CE 2.949
    Batch 9200 Train CE 2.905
    Batch 9300 Train CE 2.920
    Batch 9400 Train CE 2.988
    Batch 9500 Train CE 2.905
    Batch 9600 Train CE 2.892
    Batch 9700 Train CE 2.920
    Batch 9800 Train CE 2.919
    Batch 9900 Train CE 2.886
    Batch 10000 Train CE 2.930
    Running validation...
    Validation cross-entropy: 2.907
    Batch 10100 Train CE 2.913
    Batch 10200 Train CE 2.913
    Batch 10300 Train CE 2.904
    Batch 10400 Train CE 2.878
    Batch 10500 Train CE 2.914
    Batch 10600 Train CE 2.883
    Batch 10700 Train CE 2.865
    Batch 10800 Train CE 2.909
    Batch 10900 Train CE 2.884
    Batch 11000 Train CE 2.903
    Running validation...
    Validation cross-entropy: 2.885
    Batch 11100 Train CE 2.914
    
    Epoch 4
    Batch 11200 Train CE 2.935
    Batch 11300 Train CE 2.860
    Batch 11400 Train CE 2.823
    Batch 11500 Train CE 2.866
    Batch 11600 Train CE 2.875
    Batch 11700 Train CE 2.840
    Batch 11800 Train CE 2.840
    Batch 11900 Train CE 2.876
    Batch 12000 Train CE 2.875
    Running validation...
    Validation cross-entropy: 2.868
    Batch 12100 Train CE 2.824
    Batch 12200 Train CE 2.827
    Batch 12300 Train CE 2.849
    Batch 12400 Train CE 2.839
    Batch 12500 Train CE 2.839
    Batch 12600 Train CE 2.833
    Batch 12700 Train CE 2.835
    Batch 12800 Train CE 2.840
    Batch 12900 Train CE 2.825
    Batch 13000 Train CE 2.866
    Running validation...
    Validation cross-entropy: 2.839
    Batch 13100 Train CE 2.828
    Batch 13200 Train CE 2.803
    Batch 13300 Train CE 2.819
    Batch 13400 Train CE 2.851
    Batch 13500 Train CE 2.801
    Batch 13600 Train CE 2.797
    Batch 13700 Train CE 2.815
    Batch 13800 Train CE 2.808
    Batch 13900 Train CE 2.845
    Batch 14000 Train CE 2.799
    Running validation...
    Validation cross-entropy: 2.822
    Batch 14100 Train CE 2.842
    Batch 14200 Train CE 2.771
    Batch 14300 Train CE 2.812
    Batch 14400 Train CE 2.801
    Batch 14500 Train CE 2.805
    Batch 14600 Train CE 2.805
    Batch 14700 Train CE 2.835
    Batch 14800 Train CE 2.852
    Batch 14900 Train CE 2.800
    
    Epoch 5
    Batch 15000 Train CE 2.748
    Running validation...
    Validation cross-entropy: 2.809
    Batch 15100 Train CE 2.817
    Batch 15200 Train CE 2.788
    Batch 15300 Train CE 2.782
    Batch 15400 Train CE 2.789
    Batch 15500 Train CE 2.826
    Batch 15600 Train CE 2.792
    Batch 15700 Train CE 2.782
    Batch 15800 Train CE 2.777
    Batch 15900 Train CE 2.737
    Batch 16000 Train CE 2.779
    Running validation...
    Validation cross-entropy: 2.791
    Batch 16100 Train CE 2.781
    Batch 16200 Train CE 2.778
    Batch 16300 Train CE 2.776
    Batch 16400 Train CE 2.784
    Batch 16500 Train CE 2.748
    Batch 16600 Train CE 2.785
    Batch 16700 Train CE 2.777
    Batch 16800 Train CE 2.769
    Batch 16900 Train CE 2.761
    Batch 17000 Train CE 2.766
    Running validation...
    Validation cross-entropy: 2.788
    Batch 17100 Train CE 2.765
    Batch 17200 Train CE 2.754
    Batch 17300 Train CE 2.758
    Batch 17400 Train CE 2.780
    Batch 17500 Train CE 2.742
    Batch 17600 Train CE 2.750
    Batch 17700 Train CE 2.727
    Batch 17800 Train CE 2.777
    Batch 17900 Train CE 2.737
    Batch 18000 Train CE 2.745
    Running validation...
    Validation cross-entropy: 2.772
    Batch 18100 Train CE 2.720
    Batch 18200 Train CE 2.754
    Batch 18300 Train CE 2.737
    Batch 18400 Train CE 2.783
    Batch 18500 Train CE 2.739
    Batch 18600 Train CE 2.757
    
    Epoch 6
    Batch 18700 Train CE 2.751
    Batch 18800 Train CE 2.731
    Batch 18900 Train CE 2.717
    Batch 19000 Train CE 2.727
    Running validation...
    Validation cross-entropy: 2.772
    Validation error increasing!  Training stopped.
    
    Final training cross-entropy: 2.729
    Final validation cross-entropy: 2.772
    Final test cross-entropy: 2.775


# Part 4: Analysis

In this part, we first train a model with a 16-dimensional embedding and 128 hidden units, as discussed in the previous section;

we will use this trained model for the remainder of this section.

These methods of the Model class can be used for analyzing the model after the training is
done:


*   *display_nearest_words* lists the words whose embedding vectors are nearest to the given
word
*   *word_distance* computes the distance between the embeddings of two words
*   *predict_next_word* shows the possible next words the model considers most likely, along
with their probabilities


We also include:


*    *tsne_plot_representation* creates a 2-dimensional embedding of the distributed representation space using
an algorithm called t-SNE. (You don’t need to know what this is for the assignment, but we
may cover it later in the course.) Nearby points in this 2-D space are meant to correspond to
nearby points in the 16-D space.


```ruby
import numpy as Math

def Hbeta(D=Math.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = Math.exp(-D.copy() * beta);
    sumP = sum(P);
    H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
    P = P / sumP;
    return H, P;


def x2p(X=Math.array([]), tol=1e-5, perplexity=30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape;
    sum_X = Math.sum(Math.square(X), 1);
    D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
    P = Math.zeros((n, n));
    beta = Math.ones((n, 1));
    logU = Math.log(perplexity);

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -Math.inf;
        betamax = Math.inf;
        Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i + 1:n]))];
        (H, thisP) = Hbeta(Di, beta[i]);

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while Math.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i];
                if betamax == Math.inf or betamax == -Math.inf:
                    beta[i] = beta[i] * 2;
                else:
                    beta[i] = (beta[i] + betamax) / 2;
            else:
                betamax = beta[i];
                if betamin == Math.inf or betamin == -Math.inf:
                    beta[i] = beta[i] / 2;
                else:
                    beta[i] = (beta[i] + betamin) / 2;

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i]);
            Hdiff = H - logU;
            tries = tries + 1;

        # Set the final row of P
        P[i, Math.concatenate((Math.r_[0:i], Math.r_[i + 1:n]))] = thisP;

    # Return final P-matrix
    print("Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta)))
    return P;


def pca(X=Math.array([]), no_dims=50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape;
    X = X - Math.tile(Math.mean(X, 0), (n, 1));
    (l, M) = Math.linalg.eig(Math.dot(X.T, X));
    Y = Math.dot(X, M[:, 0:no_dims]);
    return Y;


def tsne(X=Math.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if X.dtype != "float64":
        print("Error: array X should have type float64.");
        return -1;
    # if no_dims.__class__ != "<type 'int'>":			# doesn't work yet!
    #	print("Error: number of dimensions should be an integer.");
    #	return -1;

    # Initialize variables
    X = pca(X, initial_dims);
    (n, d) = X.shape;
    max_iter = 1000;
    initial_momentum = 0.5;
    final_momentum = 0.8;
    eta = 500;
    min_gain = 0.01;
    Y = Math.random.randn(n, no_dims);
    dY = Math.zeros((n, no_dims));
    iY = Math.zeros((n, no_dims));
    gains = Math.ones((n, no_dims));

    # Compute P-values
    P = x2p(X, 1e-5, perplexity);
    P = P + Math.transpose(P);
    P = P / Math.sum(P);
    P = P * 4;  # early exaggeration
    P = Math.maximum(P, 1e-12);

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = Math.sum(Math.square(Y), 1);
        num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
        num[range(n), range(n)] = 0;
        Q = num / Math.sum(num);
        Q = Math.maximum(Q, 1e-12);

        # Compute gradient
        PQ = P - Q;
        for i in range(n):
            dY[i, :] = Math.sum(Math.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0);

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
        gains[gains < min_gain] = min_gain;
        iY = momentum * iY - eta * (gains * dY);
        Y = Y + iY;
        Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = Math.sum(P * Math.log(P / Q));
            print("Iteration ", (iter + 1), ": error is ", C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4;

    # Return solution
    return Y;

def tsne_plot_representation(model):
    """Plot a 2-D visualization of the learned representations using t-SNE."""
    print(model.params.word_embedding_weights.shape)
    mapped_X = tsne(model.params.word_embedding_weights)
    pylab.figure(figsize=(12, 9))
    for i, w in enumerate(model.vocab):
        pylab.text(mapped_X[i, 0], mapped_X[i, 1], w)
    pylab.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
    pylab.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
    pylab.show()

def tsne_plot_GLoVE_representation(W_final, b_final):
    """Plot a 2-D visualization of the learned representations using t-SNE."""
    mapped_X = tsne(W_final)
    pylab.figure(figsize=(12, 9))
    data_obj = pickle.load(open(data_location, 'rb'))
    for i, w in enumerate(data_obj['vocab']):
        pylab.text(mapped_X[i, 0], mapped_X[i, 1], w)
    pylab.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
    pylab.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
    pylab.show()

def plot_2d_GLoVE_representation(W_final, b_final):
    """Plot a 2-D visualization of the learned representations."""
    mapped_X = W_final
    pylab.figure(figsize=(12, 9))
    data_obj = pickle.load(open(data_location, 'rb'))
    for i, w in enumerate(data_obj['vocab']):
        pylab.text(mapped_X[i, 0], mapped_X[i, 1], w)
    pylab.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
    pylab.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
    pylab.show()
```

Using these methods, please answer the following questions, each of which is worth 1 point.



1.   Pick three words from the vocabulary that go well together (for example, ‘*government of united*’,‘*city of new*’, ‘*life in the*’, ‘*he is the*’ etc.).
Use the model to predict the next word.
Does the model give sensible predictions?
Try to find an example where it makes a plausible prediction even though the 4-gram wasn’t present in the dataset (*raw_sentences.txt*).
To help you out, the function *find_occurrences* lists the words that appear after a given 3-gram in the training set.

2.   Plot the 2-dimensional visualization using the method *tsne_plot_representation*.
Look at the plot and find a few clusters of related words.
What do the words in each cluster have in common?
Plot the 2-dimensional visualization using the method *tsne_plot_GLoVE_representation* for a 256 dimensional embedding.
How do the t-SNE embeddings for both models compare?
Plot the 2-dimensional visualization using the method *plot_2d_GLoVE_representation*.
How does this compare to the t-SNE embeddings?
(You don’t need to include the plots with your submission.)

3.   Are the words ‘*new*’ and ‘*york*’ close together in the learned representation?
Why or why not?

4.   Which pair of words is closer together in the learned representation: (‘*government*’, ‘*political*’), or (‘*government*’, ‘*university*’)?
Why do you think this is?

```ruby
trained_model.predict_next_word("she", "is", "a")
find_occurrences("she", "is", "a")
```

    she is a good Prob: 0.24169
    she is a part Prob: 0.07294
    she is a very Prob: 0.06827
    she is a family Prob: 0.06452
    she is a man Prob: 0.05457
    she is a big Prob: 0.05185
    she is a home Prob: 0.03534
    she is a team Prob: 0.03325
    she is a long Prob: 0.02967
    she is a new Prob: 0.02846
    The tri-gram "she is a" was followed by the following words in the training set:
        year (1 time)
        big (1 time)
        good (1 time)
1. Test on "she is a", with "part", "family" being very plausible 4 th word in that context, while not appearing in the dataset

2. Compared with the second tsne GLoVE plot which has embedded dimension of 256, the first tsne plot has 250 embedded dimension, converges with similar 110 iterations, and has smaller error of about 0.633. The first model trains much slower than GLoVE model, while this tsne graph has more distinguishable clusters, which classifies better than GLoVE model. Some example cluster of words in the first model are: ("has", "have", "had"), ("might", "will", "would", "should", "can", "may", "could"), while the second model using GLoVE groups "see" and "say" with "would", "have" with "called". The first model with neural net has more precise and clear classification result than the second model.

  For the third plot, it plots the 2d GLoVE model without tsne, while the fourth plot represents the same model with tsne, converges around 110 iteration with about 0.304 error. Compared with the third plot, the ourth plot with tsne has better clustering, while the third plot is more dispersed and distracted. 


```ruby
tsne_plot_representation(trained_model)
```

    (250, 16)
    Preprocessing the data using PCA...
    Computing pairwise distances...
    Computing P-values for point  0  of  250 ...
    Mean value of sigma:  1.0810407640085655
    Iteration  10 : error is  13.904534336872707
    Iteration  20 : error is  13.649706568082056
    Iteration  30 : error is  14.131513226855342
    Iteration  40 : error is  14.198362832757756
    Iteration  50 : error is  14.424488901617304
    Iteration  60 : error is  14.29349770066196
    Iteration  70 : error is  14.347603620287925
    Iteration  80 : error is  14.123064627096554
    Iteration  90 : error is  14.359852597874337
    Iteration  100 : error is  14.403181497189895
    Iteration  110 : error is  1.7822399276522192
    Iteration  120 : error is  1.3267981771003918
    Iteration  130 : error is  1.0994066587715356
    Iteration  140 : error is  0.9820645964050435
    Iteration  150 : error is  0.8893447064598181
    Iteration  160 : error is  0.8501889895414241
    Iteration  170 : error is  0.7967478450970407
    Iteration  180 : error is  0.7691187551110098
    Iteration  190 : error is  0.7410420921720059
    Iteration  200 : error is  0.7302200627571742
    Iteration  210 : error is  0.7236724514298175
    Iteration  220 : error is  0.7189380162974257
    Iteration  230 : error is  0.7149874762591888
    Iteration  240 : error is  0.7091406575582195
    Iteration  250 : error is  0.6976818954938886
    Iteration  260 : error is  0.6860414777218763
    Iteration  270 : error is  0.6778337226168226
    Iteration  280 : error is  0.6714217660264792
    Iteration  290 : error is  0.6665241339948741
    Iteration  300 : error is  0.6639597540918571
    Iteration  310 : error is  0.6608776865598629
    Iteration  320 : error is  0.6551361075130953
    Iteration  330 : error is  0.6458248646852011
    Iteration  340 : error is  0.6425325471402769
    Iteration  350 : error is  0.6408322530254192
    Iteration  360 : error is  0.639834673659151
    Iteration  370 : error is  0.6390956670618293
    Iteration  380 : error is  0.6380354292263417
    Iteration  390 : error is  0.6367574036958673
    Iteration  400 : error is  0.6362174398918768
    Iteration  410 : error is  0.6357387644457575
    Iteration  420 : error is  0.6355494877495865
    Iteration  430 : error is  0.6353291055419433
    Iteration  440 : error is  0.634962109051854
    Iteration  450 : error is  0.6348340537919865
    Iteration  460 : error is  0.6346869817761988
    Iteration  470 : error is  0.634354939358054
    Iteration  480 : error is  0.6341712555674262
    Iteration  490 : error is  0.634012271660342
    Iteration  500 : error is  0.6339310405660676
    Iteration  510 : error is  0.6338967734499912
    Iteration  520 : error is  0.6338774091308106
    Iteration  530 : error is  0.6338641888381421
    Iteration  540 : error is  0.6338529084642311
    Iteration  550 : error is  0.6338435896230167
    Iteration  560 : error is  0.6338356484180313
    Iteration  570 : error is  0.6338288416187464
    Iteration  580 : error is  0.633823454248764
    Iteration  590 : error is  0.6338192077900141
    Iteration  600 : error is  0.6338156463195684
    Iteration  610 : error is  0.6338126427780199
    Iteration  620 : error is  0.6338101846062707
    Iteration  630 : error is  0.6338081767469089
    Iteration  640 : error is  0.6338065033314064
    Iteration  650 : error is  0.6338050754249374
    Iteration  660 : error is  0.633803872380797
    Iteration  670 : error is  0.6338028480085196
    Iteration  680 : error is  0.6338019835025773
    Iteration  690 : error is  0.6338012757746192
    Iteration  700 : error is  0.6338006660775947
    Iteration  710 : error is  0.6338001533795595
    Iteration  720 : error is  0.6337997271390038
    Iteration  730 : error is  0.6337993530991748
    Iteration  740 : error is  0.6337990514423973
    Iteration  750 : error is  0.6337988026549337
    Iteration  760 : error is  0.6337985884653975
    Iteration  770 : error is  0.6337983949374211
    Iteration  780 : error is  0.6337982263336315
    Iteration  790 : error is  0.6337980836818287
    Iteration  800 : error is  0.6337979654647967
    Iteration  810 : error is  0.6337978650614626
    Iteration  820 : error is  0.6337977775807981
    Iteration  830 : error is  0.633797705801293
    Iteration  840 : error is  0.6337976448659564
    Iteration  850 : error is  0.6337975920897374
    Iteration  860 : error is  0.6337975468462544
    Iteration  870 : error is  0.633797508962281
    Iteration  880 : error is  0.6337974772484033
    Iteration  890 : error is  0.6337974497702629
    Iteration  900 : error is  0.6337974265000628
    Iteration  910 : error is  0.6337974076653053
    Iteration  920 : error is  0.6337973914251336
    Iteration  930 : error is  0.6337973768656655
    Iteration  940 : error is  0.6337973643741731
    Iteration  950 : error is  0.6337973537264191
    Iteration  960 : error is  0.6337973444924512
    Iteration  970 : error is  0.6337973367838915
    Iteration  980 : error is  0.6337973301302106
    Iteration  990 : error is  0.6337973244539044
    Iteration  1000 : error is  0.6337973197804718

![image-center]({{ site.url }}{{ site.baseurl }}../assets/imgs/posts/language_model_files/language_model_37_1.png){: .align-center}


```ruby
tsne_plot_GLoVE_representation(W_final, b_final)
```

    Preprocessing the data using PCA...
    Computing pairwise distances...
    Computing P-values for point  0  of  250 ...


    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:64: ComplexWarning: Casting complex values to real discards the imaginary part


    Mean value of sigma:  0.7414499269370552
    Iteration  10 : error is  17.112574187692882
    Iteration  20 : error is  15.903891159282097
    Iteration  30 : error is  17.823022390694426
    Iteration  40 : error is  17.8988129470671
    Iteration  50 : error is  17.90508793551481
    Iteration  60 : error is  18.032906039737853
    Iteration  70 : error is  17.430171652322983
    Iteration  80 : error is  17.791823417798255
    Iteration  90 : error is  17.430160829458806
    Iteration  100 : error is  17.276058343878077
    Iteration  110 : error is  2.445460592164862
    Iteration  120 : error is  1.7831495425683555
    Iteration  130 : error is  1.514816424123761
    Iteration  140 : error is  1.408098339641312
    Iteration  150 : error is  1.333816650951015
    Iteration  160 : error is  1.3042359411054738
    Iteration  170 : error is  1.2859789093274774
    Iteration  180 : error is  1.270484308076097
    Iteration  190 : error is  1.2537524682059489
    Iteration  200 : error is  1.2323294862935952
    Iteration  210 : error is  1.2241536773575152
    Iteration  220 : error is  1.2175445429163556
    Iteration  230 : error is  1.2102966054970838
    Iteration  240 : error is  1.2000799461549356
    Iteration  250 : error is  1.191544483905188
    Iteration  260 : error is  1.185410850347184
    Iteration  270 : error is  1.1818303001727737
    Iteration  280 : error is  1.1801652090564558
    Iteration  290 : error is  1.178764183355252
    Iteration  300 : error is  1.1765507062802383
    Iteration  310 : error is  1.1731327033012324
    Iteration  320 : error is  1.168540910589537
    Iteration  330 : error is  1.163721697948814
    Iteration  340 : error is  1.1584139767893855
    Iteration  350 : error is  1.1545892548493164
    Iteration  360 : error is  1.1520522885433322
    Iteration  370 : error is  1.1491463415483631
    Iteration  380 : error is  1.1451345874627616
    Iteration  390 : error is  1.1419148195690667
    Iteration  400 : error is  1.1396856431861004
    Iteration  410 : error is  1.1382239098685918
    Iteration  420 : error is  1.1370868297306882
    Iteration  430 : error is  1.1360901861311694
    Iteration  440 : error is  1.1352878443150094
    Iteration  450 : error is  1.1343275583163868
    Iteration  460 : error is  1.1333762638285985
    Iteration  470 : error is  1.1326146519223061
    Iteration  480 : error is  1.1322765476404055
    Iteration  490 : error is  1.13205157967645
    Iteration  500 : error is  1.131839593422773
    Iteration  510 : error is  1.131638875643134
    Iteration  520 : error is  1.1314530840287877
    Iteration  530 : error is  1.131270529569515
    Iteration  540 : error is  1.1310845655515285
    Iteration  550 : error is  1.1309022233834174
    Iteration  560 : error is  1.130739938002878
    Iteration  570 : error is  1.1305946712901742
    Iteration  580 : error is  1.1304752633678081
    Iteration  590 : error is  1.1303791976305713
    Iteration  600 : error is  1.1302952046570953
    Iteration  610 : error is  1.1302117310287727
    Iteration  620 : error is  1.1301083958981428
    Iteration  630 : error is  1.130024035713042
    Iteration  640 : error is  1.1299568909241
    Iteration  650 : error is  1.129899900072495
    Iteration  660 : error is  1.129851494365113
    Iteration  670 : error is  1.129809271794957
    Iteration  680 : error is  1.129771264754059
    Iteration  690 : error is  1.1297375015992641
    Iteration  700 : error is  1.129707533577688
    Iteration  710 : error is  1.129679847946885
    Iteration  720 : error is  1.1296533183685773
    Iteration  730 : error is  1.1296287179198274
    Iteration  740 : error is  1.1296064335851381
    Iteration  750 : error is  1.1295851290358387
    Iteration  760 : error is  1.1295645055182573
    Iteration  770 : error is  1.1295453873846373
    Iteration  780 : error is  1.1295271891886518
    Iteration  790 : error is  1.1295106085112372
    Iteration  800 : error is  1.129493958671409
    Iteration  810 : error is  1.1294791452271364
    Iteration  820 : error is  1.129465871629716
    Iteration  830 : error is  1.129452674629074
    Iteration  840 : error is  1.129440435770575
    Iteration  850 : error is  1.1294288640386867
    Iteration  860 : error is  1.1294174772366903
    Iteration  870 : error is  1.129406798365482
    Iteration  880 : error is  1.129396710312942
    Iteration  890 : error is  1.1293871391361658
    Iteration  900 : error is  1.1293780326641927
    Iteration  910 : error is  1.129369403095753
    Iteration  920 : error is  1.1293606927268218
    Iteration  930 : error is  1.1293529770452722
    Iteration  940 : error is  1.1293456415314356
    Iteration  950 : error is  1.1293382261185165
    Iteration  960 : error is  1.1293306477095455
    Iteration  970 : error is  1.1293233640170035
    Iteration  980 : error is  1.1293160686831774
    Iteration  990 : error is  1.1293082792737912
    Iteration  1000 : error is  1.1293001798713274


![image-center]({{ site.url }}{{ site.baseurl }}../assets/imgs/posts/language_model_files/language_model_38_3.png){: .align-center}


```ruby
plot_2d_GLoVE_representation(W_final_2d, b_final_2d)
```

![image-center]({{ site.url }}{{ site.baseurl }}../assets/imgs/posts/language_model_files/language_model_39_0.png){: .align-center}


```ruby
tsne_plot_GLoVE_representation(W_final_2d, b_final_2d)
```

    Preprocessing the data using PCA...
    Computing pairwise distances...
    Computing P-values for point  0  of  250 ...
    Mean value of sigma:  0.23915790027001896
    Iteration  10 : error is  12.573822139331822
    Iteration  20 : error is  10.52761808422471
    Iteration  30 : error is  10.230277643347483
    Iteration  40 : error is  10.201792321062891
    Iteration  50 : error is  10.26759148017883
    Iteration  60 : error is  10.329756032709255
    Iteration  70 : error is  10.409147793062388
    Iteration  80 : error is  10.365182382034577
    Iteration  90 : error is  10.415018332294085
    Iteration  100 : error is  10.378539324143624
    Iteration  110 : error is  0.8264687306050834
    Iteration  120 : error is  0.5160751571468272
    Iteration  130 : error is  0.4342438805667565
    Iteration  140 : error is  0.39366570658201167
    Iteration  150 : error is  0.3703696360627047
    Iteration  160 : error is  0.3464057819445535
    Iteration  170 : error is  0.3386322348119493
    Iteration  180 : error is  0.32567407253907854
    Iteration  190 : error is  0.3117129363030877
    Iteration  200 : error is  0.30948963900780124
    Iteration  210 : error is  0.3084495913214165
    Iteration  220 : error is  0.3077697447443233
    Iteration  230 : error is  0.3071574157603382
    Iteration  240 : error is  0.3066176276614078
    Iteration  250 : error is  0.3061637250923695
    Iteration  260 : error is  0.3057862280282505
    Iteration  270 : error is  0.30550577753683694
    Iteration  280 : error is  0.3053103985268586
    Iteration  290 : error is  0.30516129103876166
    Iteration  300 : error is  0.30503866539131214
    Iteration  310 : error is  0.3049387359318669
    Iteration  320 : error is  0.3048604082058757
    Iteration  330 : error is  0.30479614882241235
    Iteration  340 : error is  0.304743122282236
    Iteration  350 : error is  0.30470087178207694
    Iteration  360 : error is  0.3046657969700648
    Iteration  370 : error is  0.30463586616638016
    Iteration  380 : error is  0.3046104782973566
    Iteration  390 : error is  0.30458921146194545
    Iteration  400 : error is  0.30457081768756394
    Iteration  410 : error is  0.3045548122359697
    Iteration  420 : error is  0.30454110550238883
    Iteration  430 : error is  0.3045291331535103
    Iteration  440 : error is  0.3045188640363974
    Iteration  450 : error is  0.3045098763030889
    Iteration  460 : error is  0.30450186806580554
    Iteration  470 : error is  0.30449501020311454
    Iteration  480 : error is  0.304489002889391
    Iteration  490 : error is  0.30448352132749895
    Iteration  500 : error is  0.3044787607438082
    Iteration  510 : error is  0.3044745210780926
    Iteration  520 : error is  0.3044708009849443
    Iteration  530 : error is  0.30446757225607746
    Iteration  540 : error is  0.30446465988688487
    Iteration  550 : error is  0.30446201615049856
    Iteration  560 : error is  0.3044596440217502
    Iteration  570 : error is  0.3044575318090027
    Iteration  580 : error is  0.3044556063325803
    Iteration  590 : error is  0.30445389503149195
    Iteration  600 : error is  0.30445236132234726
    Iteration  610 : error is  0.3044509922299558
    Iteration  620 : error is  0.3044498155048435
    Iteration  630 : error is  0.30444869702189
    Iteration  640 : error is  0.3044476968039084
    Iteration  650 : error is  0.3044467877330923
    Iteration  660 : error is  0.3044459652524478
    Iteration  670 : error is  0.3044452236328806
    Iteration  680 : error is  0.3044445388755906
    Iteration  690 : error is  0.30444392948505306
    Iteration  700 : error is  0.3044433901440692
    Iteration  710 : error is  0.3044428926541345
    Iteration  720 : error is  0.3044424379444512
    Iteration  730 : error is  0.3044420464362049
    Iteration  740 : error is  0.30444168580467756
    Iteration  750 : error is  0.3044413498277083
    Iteration  760 : error is  0.30444104798392035
    Iteration  770 : error is  0.3044407828485272
    Iteration  780 : error is  0.3044405389157109
    Iteration  790 : error is  0.3044403190920903
    Iteration  800 : error is  0.30444011495465556
    Iteration  810 : error is  0.304439930133754
    Iteration  820 : error is  0.3044397637363192
    Iteration  830 : error is  0.3044396118081503
    Iteration  840 : error is  0.3044394751579497
    Iteration  850 : error is  0.30443934727543503
    Iteration  860 : error is  0.3044392314852308
    Iteration  870 : error is  0.3044391354398186
    Iteration  880 : error is  0.3044390442267382
    Iteration  890 : error is  0.30443895780587493
    Iteration  900 : error is  0.304438880546109
    Iteration  910 : error is  0.3044388099682113
    Iteration  920 : error is  0.3044387446893143
    Iteration  930 : error is  0.3044386863016929
    Iteration  940 : error is  0.30443863176345853
    Iteration  950 : error is  0.3044385806671761
    Iteration  960 : error is  0.3044385350993297
    Iteration  970 : error is  0.3044384937456272
    Iteration  980 : error is  0.3044384562592181
    Iteration  990 : error is  0.3044384217150869
    Iteration  1000 : error is  0.3044383911881474

![image-center]({{ site.url }}{{ site.baseurl }}../assets/imgs/posts/language_model_files/language_model_40_1.png){: .align-center}

3. As we can see, "new" and "york" are not close together, which is also as expected, because if two words close together, this means that the similarity between two words are very large. While "new" and "york" have very different meanings and nature in language, and should not be categorized into the same group.


```ruby
print(trained_model.word_distance("new", "york"))
print(trained_model.display_nearest_words("new"))
print(trained_model.display_nearest_words("york"))
```

    3.548393385127713
    old: 2.3600505246264323
    white: 2.4383030498943365
    back: 2.5759303604425643
    american: 2.6466202075516416
    such: 2.6599246057735786
    own: 2.6878098904671783
    political: 2.701124031526305
    national: 2.7494668874744193
    several: 2.7643670072967392
    federal: 2.8155523660546993
    None
    public: 0.9048826050086719
    music: 0.9105096376369168
    university: 0.9309985183745197
    city: 0.9640999583181336
    department: 0.9810341593819769
    ms.: 0.985606172603071
    john: 1.0174907854088202
    school: 1.035702631398839
    general: 1.0443388378683818
    team: 1.0641153998658832
    None


4. Comparing ("goverment", "political") and ("government", "university"), "government" is more close to "university" rather than "political". This is plausible, because both of the government and university are noun in a sentance, while political is subjective.   


```ruby
print(trained_model.word_distance("government", "political"))
print(trained_model.word_distance("government", "university"))
```

    1.4021420047033426
    0.9350013818643479

