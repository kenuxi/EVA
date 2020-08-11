from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField, SelectField, SelectMultipleField, BooleanField, DecimalField
from wtforms.validators import DataRequired, NumberRange


class SelectFileForm(FlaskForm):
    file = SelectField(label='.csv in data/')
    file_submit = SubmitField(label='Submit')


class UploadForm(FlaskForm):
    csv_file = FileField(label='CSV', validators=[DataRequired(), FileAllowed(['csv'], 'Wrong file format!')])
    csv_submit = SubmitField(label='Upload')


class LabelForm(FlaskForm):
    label_column = SelectField(label='Label Column', choices=[])

    ratio_bool = BooleanField(label='Ratio_bool')
    ratio = DecimalField(label='Ratio', default=1, rounding=1,
                         validators=[DataRequired(), NumberRange(min=0.1, max=99.9, message='lol')])

    normalize_bool = BooleanField(label='normalize_bool')

    preprocess_bool = BooleanField(label='preprocess?')
    preprocess = DecimalField(label='Dimensions', default=2, rounding=1,
                              validators=[DataRequired(), NumberRange(min=1, max=100, message='lol')])

    inliers = SelectMultipleField(label='Inlier Data', choices=[])
    outliers = SelectMultipleField(label='Outlier Data', choices=[])

    label_submit = SubmitField(label='Submit')


class VisForm(FlaskForm):

    PCA1 = BooleanField(label='scatter', description='scatter')
    PCA2 = BooleanField(label='box', description='box')
    PCA3 = BooleanField(label='k', description='k')
    PCA4 = BooleanField(label='dendo', description='dendrogram')
    PCA5 = BooleanField(label='density', description='density')
    PCA6 = BooleanField(label='heat', description='heat')

    LLE1 = BooleanField(label='scatter', description='scatter')
    LLE2 = BooleanField(label='box', description='box')
    LLE3 = BooleanField(label='k', description='k')
    LLE4 = BooleanField(label='dendo', description='dendrogram')
    LLE5 = BooleanField(label='density', description='density')
    LLE6 = BooleanField(label='heat', description='heat')

    TSNE1 = BooleanField(label='scatter', description='scatter')
    TSNE2 = BooleanField(label='box', description='box')
    TSNE3 = BooleanField(label='k', description='k')
    TSNE4 = BooleanField(label='dendo', description='dendrogram')
    TSNE5 = BooleanField(label='density', description='density')
    TSNE6 = BooleanField(label='heat', description='heat')

    UMAP1 = BooleanField(label='scatter', description='scatter')
    UMAP2 = BooleanField(label='box', description='box')
    UMAP3 = BooleanField(label='k', description='k')
    UMAP4 = BooleanField(label='dendo', description='dendrogram')
    UMAP5 = BooleanField(label='density', description='density')
    UMAP6 = BooleanField(label='heat', description='heat')

    ISOMAP1 = BooleanField(label='scatter', description='scatter')
    ISOMAP2 = BooleanField(label='box', description='box')
    ISOMAP3 = BooleanField(label='k', description='k')
    ISOMAP4 = BooleanField(label='dendo', description='dendrogram')
    ISOMAP5 = BooleanField(label='density', description='density')
    ISOMAP6 = BooleanField(label='heat', description='heat')

    KMAP1 = BooleanField(label='scatter', description='scatter')
    KMAP2 = BooleanField(label='box', description='box')
    KMAP3 = BooleanField(label='k', description='k')
    KMAP4 = BooleanField(label='dendo', description='dendrogram')
    KMAP5 = BooleanField(label='density', description='density')
    KMAP6 = BooleanField(label='heat', description='heat')

    MDS1 = BooleanField(label='scatter', description='scatter')
    MDS2 = BooleanField(label='box', description='box')
    MDS3 = BooleanField(label='k', description='k')
    MDS4 = BooleanField(label='dendo', description='dendrogram')
    MDS5 = BooleanField(label='density', description='density')
    MDS6 = BooleanField(label='heat', description='heat')

    vis_submit = SubmitField(label='Submit')

