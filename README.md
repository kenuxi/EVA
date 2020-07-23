
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
##### Info 
Principal component analysis (PCA) is a classic algorithm for dimensionality reduction. 
PCA transforms points from the original space to the space of uncorelated features over given dataset
via eigendecomposition of covariance matrix.
PCA by design is a linear algorithm meaning that it's not capable of capturing non-linear correlations.

https://en.wikipedia.org/wiki/Principal_component_analysis

##### Adjustable parameters
* n_dim - change the dimensionality of the projection
### LLE
##### Info
LLE is a non-linear dimensionality reduction algorithms. LLE is a two step procedure. The first step consists 
of finding k-nearest neighbours of each point and computing the weights of reconstructing original point with 
it's neighbours. Second step embeds high-dimensional data points in lower dimensional space using weights learned 
in the first step.

https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf

###### Adjustable parameters
* n_dim - change the dimensionality of the projection
* k_neighbours - change number of neighbours used to reconstruct the point in the first step
### T-SNE

### UMAP
### ISOMAP
### k-MAPPER
### MDS
