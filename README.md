# EVA

Vision is arguably one of our greatest strengths as humans. Even the most complex ideas tend to become much easier to understand as soon as we find a way to visualize them. Therefore, when we are working with datasets that have more than 3 dimensions, it can be challenging just to understand what’s going on inside of them.

Anomaly detection is a subfield of AI that detects anomalies in datasets. A
machine learning algorithm learns the "patterns" that the majority of cases
adhere to and then singles out the few cases that deviate from those normal
patterns. Appropriate projections of the high-dimensional datasets to two
dimensions often help by isolating anomalies in the image of those projections.
The anomalies can then easily be spotted by a human just by looking at the
image. There are many different techniques with different strengths and
weaknesses.

We have developed develop a modular, extensible anomaly visualization
framework with graphical user interface that allows to evaluate different data
visualizations techniques on user provided datasets.

# Installing
Create new python3 virtual environment:
```
python3 -m venv venv
```
and source it:
```
source venv/bin/activate
```  
Install libraries:
```
pip install -r requirements.txt
```
Run the app:
```
python app.py
```
Go to this address in your web browser:
```
http://127.0.0.1:5000
```

# Docker
Create docker image
```
docker build -t name:tag . (example: docker build -t EVA:latest .)
```
Run the app
```
docker run -p 5000:port name:tag (example: docker run -p 5000:5000 EVA:latest)
```
# Documentation

## Usage

After you run the server on `http://127.0.0.1:5000`, you will be greeted with this home screen.

![home](/images/home.png)

Here, you can either choose an existing dataset that are in `/data` or you can upload your own dataset. After, you upload your own dataset it gets listed on the dropdown where you can choose and submit it. After you decide to submit the dataset, a table will be created for you to have a peek into it.

![table](/images/table.png)

Now, you can apply filter to your dataset and also selet inliers/outliers. You can normalise and preprocess data and also select the ratio for the outliers.

![filter](/images/filter.png)

Now for the main part, you will have the oppertunity to choose an algorithm to apply to the dataset as well as a visualisation technique you prefer.

For algorithm, you can choose between:

* PCA
* LLE
* TSNE
* UMAP
* ISOMAP
* K-Mapper
* MDS

As visualisation methods, you can choose between:

* Scatter plot
* Box plot
* k-nearest neighbour
* Dendogram
* Density plot
* Heatmap

![pick](/images/picker.png)

After you choose algorithms and visualisations methods, you will be redirected to the page which shows the plots obtained from chosen algorithms

## Dimensionality reduction algorithms
Dimensionality reduction algorithms help with understanding data through visualisation.
The main concept of dimensionality reduction is using techniques to embed high dimensional (D > 3) 
points to lower, usually 2 or 3 dimensional space to plot the data.

EVA implements following dimensionality reduction algorithms:
### PCA
#### Info 
Principal component analysis (PCA) is a classic algorithm for dimensionality reduction. 
PCA transforms points from the original space to the space of uncorrelated features over given dataset
via eigendecomposition of covariance matrix.
PCA by design is a linear algorithm meaning that it's not capable of capturing non-linear correlations.

https://en.wikipedia.org/wiki/Principal_component_analysis

#### Adjustable parameters
* n_dim - change the dimensionality of the projection
### LLE
#### Info
Locally Linear Embedding (LLE) is a nonlinear dimensionality reduction algorithm. LLE is a two step procedure. The first step consists 
of finding k-nearest neighbours of each point and computing the weights of reconstructing original point with 
it's neighbours. Second step embeds high-dimensional data points in lower dimensional space using weights learned 
in the first step.

https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf

#### Adjustable parameters
* n_dim - change the dimensionality of the projection
* k_neighbours - change number of neighbours used to reconstruct the point in the first step
### t-SNE
#### Info
T-distributed Stochastic Neighbour Embedding (t-SNE) is a nonlinear dimensionality reduction
algorithm. In order to create low-dimensional mapping, t-SNE computes similarity of high-dimensional
points through creating a probability distribution. This distribution is used for creating low-dimensional
mapping of points by comparing it to the distribution of low-dimensional points and minimizing
the difference between those two using K-L Divergence.

http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

#### Adjustable parameters
* n_dim - change the dimensionality of the projection
* perplexity - change the target perplexity of the algorithm, balance attention between global and local structure of the data

### UMAP
#### Info
Uniform Manifold Approximation and Projection (UMAP) is a general dimensionality reduction algorithm using topological tools.
UMAP assumes that the data is uniformly distributed on a Rimannian manifold which is locally connected and the Rimannian metric is locally constant or can be approximated as such.
UMAPS models data manifold with fuzzy topological structure and embed the data into low dimensional space
by finding closest possible equivalent topological structure. 

https://arxiv.org/pdf/1802.03426.pdf

#### Adjustable parameters 
* n_dim - change the dimensionality of the projection
* k_neighbours - change the number of neighbours used in creation of k-neighbour graph 
### ISOMAP
#### Info
ISOMAP is a nonlinear dimensionality reduction algorithm. In order to create low dimensional embedding of data ISOMAP
creates weighted graph of k-nearest neighbours with euclidean distance as weights of the graph. After the graph is 
created, ISOMAP computes shortest paths between two nodes and use this information to create low dimensional representation with MDS algorithm.

http://web.mit.edu/cocosci/Papers/sci_reprint.pdf
#### Adjustable parameters 
* n_dim - change the dimensionality of the projection
* k_neighbours - change the number of neighbours used in creation of k-neighbour graph
### k-MAPPER

#### Info
Kepler-MAPPER (k-MAPPER) is python library implementing MAPPER algorithm form the topological data analysis field.
k-MAPPER use embedding created by other dimensionality reduction algorithm (ex. t-SNE) and pass it to MAPPER algorithm.

https://github.com/scikit-tda/kepler-mapper
#### Adjustable parameters 
* n_dim - change the dimensionality of the projection
* k_neighbours - change the number of neighbours passed to the MAPPER algorithm
### MDS
#### Info
Multidimensional Scaling (MDS) is a nonlinear dimensionality reduction algorithm.
Low dimensional embedding is obtained by solving eigenvalue problem in double centered matrix of squared proximity matrix.

https://en.wikipedia.org/wiki/Multidimensional_scaling
#### Adjustable parameters 
* n_dim - change the dimensionality of the projection

### Extending EVA to new dimensionality reduction algorithm

Structure of this project allows extending dashboard to support new dimensionality
reduction algorithms with custom, controllable parameters.
In order to add new algorithm, follow these steps:

1. Extend `EvaData` class in `application/plotlydash/Dashboard.py` with `apply_{name_of_your_alg}` method following convention of the other `apply` methods.
2. Extend `_getgraph` and `_getdropdowns` methods in `DimRedDash` class in `application/plotldydash/dim_red_dshboards.py` to support new plots and callbacks for your new algorithm.
3. Add new form options to `VisForm` class in `forms.py` file,
4. Edit form code in `application/templates/home.html` to include your new, updated form.

# Results

To demonstrate the results we have selected 3 datasets, namely, FMNIST, MNIST and FishBowl. We have then plotted 2D scatter-plot for all the datasets and selected algorithms.

## FMNIST

Fashion-MNIST (FMNIST) is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

https://github.com/zalandoresearch/fashion-mnist

This is how the data looks (each class takes three-rows):

![fmnist](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

For the visualisation purpose here, we have applied following filters:

* Shoe images as the Inliers  (6000 Images)
* T-shirt images as the Outliers (6000 Images)
* Define 1% outliers in the dataset -> 60 T-shirt images
* Normalise: True
* PCA preposcessed: True


### PCA
![pca](/images/Fmnist/pca.png)

### LLE
* k-neighbour: 20

![lle](/images/Fmnist/lle.png)

### TSNE
* perplexity: 30

![tsne](/images/Fmnist/tsne.png)

### UMAP
* k-neighbour: 15

![umap](/images/Fmnist/umap.png)

### ISOMAP
* k-neighbour: 20

![isomap](/images/Fmnist/isomap.png)


From the visualisation it is clear that `UMAP` performs the better job of isolating the outliers.

## MNIST

The Modified National Institute of Standards and Technology (MNIST) database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

http://yann.lecun.com/exdb/mnist/

Sample image for MNIST dataset:

![MNIST](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/220px-MnistExamples.png)

For the visualisation purpose here, we have applied following filters:

* Zeros Inliers: 6742 images
* Ones Outliers : 67 images (1%)
* Normalise: True
* PCA preposcessed: True

### PCA
![pca](/images/mnist/pca.png)

### LLE
* k-neighbour: 5

![lle](/images/mnist/lle.png)


### UMAP
* K-neighbour: 15

![umap](/images/mnist/umap.png)

From the visualisation it is clear that `UMAP` performs the better job of isolating the outliers.


## FishBowl

Fish Bowl dataset comprises a sphere embedded in 3D whose top cap has been removed. In other words, it is a punctured sphere, which is sparsely sampled at the bottom and densely at the top.

For the visualisation purpose here, we have applied following filters:

* Rows: 2010
* Inliers: 2000
* Outliers: 10

* Normalise: False
* PCA preposcessed: False

### PCA
![pca](/images/Fishbowl/pca.png)

### LLE
* k-neighbour: 5

![lle](/images/Fishbowl/lle.png)

### TSNE
* perplexity: 30

![tsne](/images/Fishbowl/tsne.png)

### UMAP
* k-neighbour: 20

![umap](/images/Fishbowl/umap.png)

### ISOMAP
* k-neighbour: 20

![isomap](/images/Fishbowl/isomap.png)

### MDS
![mds](/images/Fishbowl/mds.png)

From the visualisation, we can say that `LLE` argueably performs the better job of isolating the outliers.
