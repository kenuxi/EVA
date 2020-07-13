from werkzeug.serving import run_simple
from flask import Flask, render_template, url_for, redirect, flash, request, jsonify
from flask_uploads import configure_uploads, UploadSet
from forms import SelectFileForm, UploadForm, VisForm, LabelForm
from config import app_secret_key, session
from statistics_methods import DataStatistics
import pandas as pd
import os
from config import alg_types


to_reload = False


def get_app():
    app = Flask(__name__, instance_relative_config=False,
                template_folder='application/templates')
    app.config['SECRET_KEY'] = app_secret_key
    csv_files = UploadSet('data', ('csv',), default_dest=lambda x: 'data')
    configure_uploads(app, csv_files)

    @app.route('/', methods=['GET', 'POST'])
    @app.route('/home', methods=['GET', 'POST'])
    def home():
        ds = DataStatistics()

        file_form = SelectFileForm()
        up_form = UploadForm()
        vis_form = VisForm()
        label_form = LabelForm()
        if up_form.csv_submit.data and up_form.validate_on_submit():
            csv_data = up_form.csv_file.data
            filename = csv_data.filename
            if filename in os.listdir(os.path.join('data')):
                flash('Filename exists!', 'danger')
                return redirect(url_for('home'))
            csv_data.save(os.path.join('data', filename))
            flash('Your file has been Added!', 'success')
            return redirect(url_for('home'))

        elif file_form.file_submit.data and file_form.validate_on_submit():
            # loading data from file into wrapper class
            ds.load_data(file_form.file.data)
            # separate dataframe for easier access
            session['df'] = ds.pandas_data_frame
            # populating label choices with data from file
            label_columns = [(str(col), str(col)) for col in ds.pandas_data_frame.columns]
            label_columns.reverse() #reverse cause last col is usually label
            label_form.label_column.choices = label_columns

            # populating inlier and outlier choices with initial options
            # this changes dynamically once in app
            initial_labels = [(str(col), str(col)) for col in session['df'][label_columns[0][0]].unique()]
            label_form.inliers.choices = initial_labels
            label_form.outliers.choices = initial_labels
            return render_template('home.html', title='Home',
                                   df=ds.pandas_data_frame,
                                   file_form=file_form,
                                   up_form=up_form,
                                   label_form=label_form)

        elif label_form.submit.data:
            ds.label_column = label_form.label_column.data
            ds.inlier = label_form.outliers.data
            ds.outliers = label_form.inliers.data

            return render_template('home.html', title='Home',
                                   df=session['df'],
                                   file_form=file_form,
                                   up_form=up_form,
                                   label_form=label_form,
                                   vis_form=vis_form)           # f'{label_form.label_column.data}, {label_form.outliers.data}, {label_form.inliers.data}'

        elif vis_form.submit.data:
            dashboard_config = {'location': session['filename'],
                                'target': vis_form.target.data,
                                'PCA': [],
                                'LLE': [],
                                'TSNE': [],
                                'UMAP': [],
                                'ISOMAP': []
                                }

            for alg in alg_types:
                for field in vis_form:
                    if field.type == "BooleanField" and alg in field.short_name:
                        if field.data:
                            dashboard_config[alg].append(field.description)

            session['dashboard_config'] = dashboard_config
            print(dashboard_config)
            return redirect(url_for('reload'))      # f"{session['dashboard_config']}"

        return render_template('home.html', title='Home', file_form=file_form, up_form=up_form)

    @app.route('/getlabels/<column>', methods=['GET'])
    def getlabel(column):
        # df = request.args.get('df')
        # column = request.args.get('column')
        labels = [label for label in session['df'][column].unique()]
        return jsonify({'labels': labels})

    @app.route('/reload')
    def reload():
        global to_reload
        to_reload = True
        return redirect('/dashboard')

    if to_reload:
        with app.app_context():
            from application.plotlydash.Dashboard_new import FileDashboard
            app = FileDashboard(app).create_dashboard(session['dashboard_config'])

    return app


class AppReloader:
    def __init__(self, create_app):
        self.create_app = create_app
        self.app = create_app()

    def get_application(self):
        global to_reload
        if to_reload:
            self.app = self.create_app()
            to_reload = False
        return self.app

    def __call__(self, environ, start_response):
        app = self.get_application()
        return app(environ, start_response)


application = AppReloader(get_app)
if __name__ == "__main__":
    run_simple(hostname='localhost', port=5000, application=application,
               use_reloader=True, use_debugger=True, use_evalex=True)

