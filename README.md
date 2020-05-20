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