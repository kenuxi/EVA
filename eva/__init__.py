import os
from flask import Flask, render_template, url_for, redirect, flash, jsonify, request
from flask_uploads import configure_uploads, UploadSet
from eva.forms import SelectFileForm, UploadForm, LabelForm, VisForm
from eva.config import app_secret_key, session, alg_types
from eva.statistics_methods import DataStatistics


to_reload = False


def get_app():
    app = Flask(__name__, instance_relative_config=False)
    app.config['SECRET_KEY'] = app_secret_key
    csv_files = UploadSet('data', ('csv',), default_dest=lambda x: 'data')
    configure_uploads(app, csv_files)

    @app.route('/', methods=['GET', 'POST'])
    @app.route('/home', methods=['GET', 'POST'])
    def home():
        '''Serves homepage with initial forms. On submit the data is validated and processed.'''
        file_form = SelectFileForm()
        up_form = UploadForm()
        vis_form = VisForm()

        if up_form.csv_submit.data and up_form.validate_on_submit():
            '''Upload submitted. If filename does not exist, file is saved to 'eva/data' directory.'''
            csv_data = up_form.csv_file.data
            filename = csv_data.filename
            if filename in os.listdir(os.path.join('eva/data')):
                flash('Filename exists!', 'danger')
                return redirect(url_for('home'))

            csv_data.save(os.path.join('eva/data', filename))
            flash('Your file has been Added!', 'success')
            return redirect(url_for('home'))

        if vis_form.data and vis_form.validate_on_submit():
            dashboard_config = {'ds': session['ds'],
                                'PCA': [],
                                'LLE': [],
                                'TSNE': [],
                                'UMAP': [],
                                'ISOMAP': [],
                                'KMAP': [],
                                'MDS': [],
                                }

            for alg in alg_types:
                for field in vis_form:
                    if field.type == "BooleanField" and alg in field.short_name:
                        if field.data:
                            dashboard_config[alg].append(field.description)

            session['dashboard_config'] = dashboard_config
            if not session['ds'].label_column:
                session['ds'].label_column = 'None'
            session['ds'].create_unlabeled_df()

            return redirect(url_for('reload'))  # f"{session['dashboard_config']}"

        return render_template('home.html', title='Home', file_form=file_form, up_form=up_form, vis_form=vis_form)

    @app.route('/get_label_form/', methods=['GET'])
    def get_label_form():
        file = request.args.get('file', 0, type=str)

        session['ds'] = DataStatistics()    # initiated wrapper class
        session['ds'].load_data(file)   # loading file into wrapper class

        # separate dataframe for easier access
        session['df'] = session['ds'].pandas_data_frame

        label_form = LabelForm()
        # populating label choices with data from file
        label_columns = [(str(col), str(col)) for col in session['df']]
        label_columns.append((None, 'None'))
        label_columns.reverse()     # reverse cause last col is usually label
        label_form.label_column.choices = label_columns
        # keeping track of selected label_column in backend
        session['ds'].label_column = label_columns[0][0]

        return jsonify({'result': render_template('label_form.html',
                                                  df=session['ds'].pandas_data_frame,
                                                  label_form=label_form)})

    @app.route('/post_label/<label>', methods=['GET'])
    def post_label(label):
        session['ds'].label_column = label
        return jsonify({'label_column': label})

    @app.route('/getfiles/', methods=['GET'])
    def getfiles():
        files = [('eva/data/' + file, file) for file in os.listdir('eva/data') if '.csv' in file]
        return jsonify({'files': sorted(files)})

    @app.route('/reload')
    def reload():
        global to_reload
        to_reload = True
        return redirect('/dashboard')

    if to_reload:
        with app.app_context():
            from eva.plotlydash.Dashboard_new import FileDashboard
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



