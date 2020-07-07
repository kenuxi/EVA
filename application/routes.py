from flask import current_app as app
from flask import Flask, render_template, url_for, flash, redirect


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template('home.html', title='Home', form=form)
