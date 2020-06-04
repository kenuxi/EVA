from flask import Flask
from flask import current_app as app
from flask import Flask, render_template, url_for, flash, redirect
from forms import HomePageForm
from config import app_secret_key

def app_base(debug=True):
    app = Flask(__name__, instance_relative_config=False)
    app.config['SECRET_KEY'] = app_secret_key

    @app.route('/')
    def home():
        form = HomePageForm()
        return render_template('home.html', title='Home', form=form)

    @app.route("/about")
    def about():
        return render_template('about.html', title='About')

    @app.route('/dashboard')
    def dashboard():
        return 'Please specify parameters on the homepage.'

    return app
