from flask import current_app as app
from flask import Flask, render_template, url_for, flash, redirect
from forms import RegistrationForm, LoginForm, FileSelectForm


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    form = FileSelectForm()
    return render_template('home.html', title='Home', form=form)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    # if form.validate_on_submit():
    #    flash('Logged in!', 'success')
    #    return redirect(url_for('home')
    # else:
    # flash('Login Unsuccessful. Pleas try again., 'danger')
    return render_template('login.html', title='Login', form=form)


