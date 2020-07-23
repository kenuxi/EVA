
# EVA

## your overall plan
- Project definition: 

### Interactive data visualization for anomaly detection

Anomaly detection is a subfield of AI that detects anomalies in datasets. A
machine learning algorithm learns the "patterns" that the majority of cases
adhere to and then singles out the few cases that deviate from those normal
patterns. Appropriate projections of the high-dimensional datasets to two
dimensions often help by isolating anomalies in the image of those projections.
The anomalies can then easily be spotted by a human just by looking at the
image. There are many different techniques with different strengths and
weaknesses.
Your job would be to develop a modular, extensible anomaly visualization
framework with graphical user interface that allows to evaluate different data
visualizations techniques on user provided datasets. Time series datasets
should be paid particular attention to.


### implementation
- Flask + Plotly Dash

### Requirements firts Prototype 
- Upload a CSV
- Configuration Screen
- Data preview
- Run PCA & T-SNE
- Visualise Plotly Dash


## your project management 
### Tasks
- Generating Anomalines to CSV files      ->Rocco
- Handling PCA application to Datasets    ->Alberto
- Build Flask Backend and Homepage        ->Kenneth
- Start implementing Docker               ->Rocco
- Git & Team Management                   ->Uttam
- T-SNE implementation                    ->Alberto
- Plot from uploaded CSV                  ->Uttam


## Data
- IRIS dataset changed to have anomaly
- MNIST dataset changed to have anomaly

## splitting up the requirements into tasks (at least the first ones)

## assignment of exactly one person to each task

Week 1: Research 

Week 2: More specific research into Visualisation Frameworks -> Define Milestones

Week 3: Create First demo prototype using sample data

Week 4-5: Running first prototype, including Flask, applying at least T-SNE, PCA

=======
# dash_iris_demo_flask
This code shows how to create interactive plots using dash using iris dataset
fetched from remote source. This time, plotly dash is hosted on a Flask app allowing
to develop our app further.
You can check the details here:
```
https://hackersandslackers.com/plotly-dash-with-flask/
```
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
Run example:
```
python app.py
```
Iris example is at:
```
http://127.0.0.1:5000/iris_example/
```

# Docker
Create docker image
```
docker build -t name:tag . (example: docker build -t iris-app:latest .)
```
Run the app
```
docker run -p 5000:port name:tag (example: docker run -p 5000:5000 iris-app:latest)
```
# Documentation

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
5. Add new form options to `VisForm` class in `forms.py` file,
6. Edit form code in `application/templates/home.html` to include your new, updated form.



