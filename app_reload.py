import os
from werkzeug.serving import run_simple
from flask import Flask, render_template, url_for, redirect, flash, jsonify, request
from flask_uploads import configure_uploads, UploadSet
from forms import SelectFileForm, UploadForm, VisForm, LabelForm
from config import app_secret_key, session
from statistics_methods import DataStatistics
from config import alg_types


to_reload = False


def get_app():
    app = Flask(__name__, instance_relative_config=False,
                template_folder='application/templates')
    app.config['SECRET_KEY'] = app_secret_key
    csv_files = UploadSet('data', ('csv',), default_dest=lambda x: 'data')
    configure_uploads(app, csv_files)
    session['ds'] = DataStatistics()

    @app.route('/', methods=['GET', 'POST'])
    @app.route('/home', methods=['GET', 'POST'])
    def home():
        file_form = SelectFileForm()
        up_form = UploadForm()
        label_form = LabelForm()
        vis_form = VisForm()

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
            session['ds'].load_data(file_form.file.data)

            # separate dataframe for easier access
            session['df'] = session['ds'].pandas_data_frame

            # populating label choices with data from file
            label_columns = [(str(col), str(col)) for col in session['df']]
            label_columns.reverse()     # reverse cause last col is usually label
            label_form.label_column.choices = label_columns
            # keeping track of selected label_column in backend
            session['ds'].label_column = label_columns[0][0]

            # populating inlier and outlier choices with initial options
            # this changes dynamically once in app
            initial_labels = [(str(col), str(col)) for col in session['df'][label_columns[0][0]].unique()]
            initial_labels.sort()
            label_form.inliers.choices = initial_labels
            label_form.outliers.choices = initial_labels

            return render_template('home.html', title='Home',
                                   df=session['ds'].pandas_data_frame,
                                   file_form=file_form,
                                   up_form=up_form,
                                   label_form=label_form)

        elif label_form.label_submit.data:
            session['ds'].label_column = label_form.label_column.data
            session['ds'].inliers = label_form.inliers.data
            session['ds'].outliers = [outlier for outlier in label_form.outliers.data if outlier not in session['ds'].inliers]
            session['ds'].ratio = label_form.ratio.data if label_form.ratio_bool.data else None
            session['ds'].normalize = label_form.normalize_bool.data
            session['ds'].pre_process = label_form.preprocess.data if label_form.preprocess_bool.data else None
            session['ds'].create_labeled_df()

            # populating label choices with data from file
            label_columns = [(str(col), str(col)) for col in session['df']]
            label_columns.reverse()     # reverse cause last col is usually label
            label_form.label_column.choices = label_columns

            # populating inlier and outlier choices with initial options
            # this changes dynamically once in app
            initial_labels = [(str(col), str(col)) for col in session['df'][label_columns[0][0]].unique()]
            initial_labels.sort()
            label_form.inliers.choices = initial_labels
            label_form.outliers.choices = initial_labels

            print(session['ds'].inliers, session['ds'].outliers)
            return render_template('home.html', title='Home',
                                   df=session['ds'].pandas_data_frame,
                                   file_form=file_form,
                                   up_form=up_form,
                                   label_form=label_form,
                                   vis_form=vis_form)

        elif vis_form.vis_submit.data:
            print(vis_form.validate())
            dashboard_config = {'ds': session['ds'],
                                'PCA':  [],
                                'LLE':  [],
                                'TSNE': [],
                                'UMAP': [],
                                'ISOMAP': [],
                                'KMAP': [],
                                'MDS':  [],
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
    def getlabels(column):
        labels = [str(label) for label in session['ds'].pandas_data_frame[str(column)].unique()]
        session['ds'].label_column = column
        return jsonify({'labels': labels})

    @app.route('/getnumrows/')
    @app.route('/getnumrows/<selected_labels>', methods=['GET'])
    def getnumrows(selected_labels=None):
        if not selected_labels:
            return jsonify({'num_rows': 0})

        labels_list = selected_labels.split(',')
        count = 0
        for label in labels_list:
            count += (session['df'][session['ds'].label_column] == label).sum()
        return jsonify({'num_rows': int(count)})

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

