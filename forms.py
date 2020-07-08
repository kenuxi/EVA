import os
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField, SelectField, SelectMultipleField, BooleanField, widgets
from wtforms.validators import DataRequired
from config import vis_types


class SelectFileForm(FlaskForm):
    files = [('data/' + file, file) for file in os.listdir('data') if '.csv' in file]
    file = SelectField(label='.csv in data/', choices=files)
    file_submit = SubmitField(label='Submit')


class UploadForm(FlaskForm):
    csv_file = FileField(label='CSV', validators=[DataRequired(), FileAllowed(['csv'], 'Wrong file format!')])
    csv_submit = SubmitField(label='Upload')


class VisForm(FlaskForm):
    target = SelectField(label='Label Column', choices=[])

    PCA1 = BooleanField(label='scatter', description='scatter')
    PCA2 = BooleanField(label='box', description='box')
    PCA3 = BooleanField(label='k', description='k')
    PCA4 = BooleanField(label='dendo', description='dendogram')

    LLE1 = BooleanField(label='scatter', description='scatter')
    LLE2 = BooleanField(label='box', description='box')
    LLE3 = BooleanField(label='k', description='k')
    LLE4 = BooleanField(label='dendo', description='dendogram')

    TSNE1 = BooleanField(label='scatter', description='scatter')
    TSNE2 = BooleanField(label='box', description='box')
    TSNE3 = BooleanField(label='k', description='k')
    TSNE4 = BooleanField(label='dendo', description='dendogram')

    UMAP1 = BooleanField(label='scatter', description='scatter')
    UMAP2 = BooleanField(label='box', description='box')
    UMAP3 = BooleanField(label='k', description='k')
    UMAP4 = BooleanField(label='dendo', description='dendogram')

    ISOMAP1 = BooleanField(label='scatter', description='scatter')
    ISOMAP2 = BooleanField(label='box', description='box')
    ISOMAP3 = BooleanField(label='k', description='k')
    ISOMAP4 = BooleanField(label='dendo', description='dendogram')

    submit = SubmitField(label='Submit')

