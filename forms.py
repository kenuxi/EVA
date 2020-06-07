import os
from flask_wtf import FlaskForm
from wtforms import SubmitField, BooleanField, SelectField
# from wtforms import StringField, PasswordField
# from wtforms.validators import DataRequired, Length, Email, EqualTo

class HomePageForm(FlaskForm):
    files = [file for file in os.listdir('application/data') if '.csv' in file]
    select = SelectField(label='Select from ' + os.getcwd(), choices=files)
    pca = BooleanField('PCA')
    tsne = BooleanField('T-SNE')
    submit = SubmitField(label='Submit')

'''
These are not needed at the moment.

class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField(label='Sign Up')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField(label='Log in')


class FileSelectForm(FlaskForm):
    files = [file for file in os.listdir(app.root_path + '/data') if '.csv' in file]
    select = SelectField(label='Select from ' + os.getcwd(), choices=files)
    submit = SubmitField(label='Submit')

'''
