import os
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField, SelectField, SelectMultipleField, BooleanField, widgets
from wtforms.validators import DataRequired
from config import vis_types
# from wtforms import PasswordField
# from wtforms.validators import Length, Email, EqualTo


class SelectFileForm(FlaskForm):
    files = [('data/' + file, file) for file in os.listdir('data') if '.csv' in file]
    file = SelectField(label='.csv in data/', choices=files)
    file_submit = SubmitField(label='Submit')


class AlgorithmForm(FlaskForm):
    submit = SubmitField(label='Submit')


class UploadForm(FlaskForm):
    csv_file = FileField(label='CSV', validators=[DataRequired(), FileAllowed(['csv'], 'Wrong file format!')])
    csv_submit = SubmitField(label='Upload')


class VisForm(FlaskForm):
    target = SelectField(label='Label Column', choices=[])
    PCA1 = BooleanField(label='scatter', description="scatter")
    PCA2 = BooleanField(label='box', description="box")
    PCA3 = BooleanField(label='k', description="k")
    LLE1 = BooleanField(label='scatter', description="scatter")
    LLE2 = BooleanField(label='box', description="box")
    LLE3 = BooleanField(label='k', description="k")
    TSNE1 = BooleanField(label='scatter', description="scatter")
    TSNE2 = BooleanField(label='box', description="box")
    TSNE3 = BooleanField(label='k', description="k")
    submit = SubmitField(label='Submit')


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()


class MultiAlgForm(FlaskForm):
    pca = MultiCheckboxField(label='PCA')
    lle = MultiCheckboxField(label='LLE')
    tsne = MultiCheckboxField(label='TSNE')
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
