from werkzeug.serving import run_simple
from flask import Flask, render_template, url_for, redirect, request, flash
from flask_uploads import configure_uploads, UploadSet
from forms import SelectFileForm, ALgorithmForm, UploadForm
from config import app_secret_key, session
from stats import DataStats

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
        file_form = SelectFileForm()
        alg_form = ALgorithmForm()
        if request.method == 'POST':
            ds = DataStats(file_form.file.data)
            ds.load_data()
            session['location'] = ds.file_name
            shape = ds.DF.shape     # This is now the pd.dataframe.shape...
            return render_template('home.html', title='Home', shape=shape, df=ds.DF,
                                   file_form=file_form,
                                   alg_form=alg_form)

        elif alg_form.submit.data and alg_form.validate_on_submit():
            dashboard_config = {
                'target': alg_form.target.data,
                'algorithm': alg_form.algorithm.data
                }

            session['dashboard_config'] = dashboard_config
            return redirect(url_for('reload'))
        return render_template('home.html', title='Home', file_form=file_form)

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        form = UploadForm()
        if form.validate_on_submit():
            csv_data = form.csv_file.data
            print(csv_data)
            print(form.csv_file.name)
            filename = csv_files.save(csv_data)
            flash('Your file has been Added!', 'success')
            return redirect(url_for('home'))
            # return f'Filename: {filename}'
            # flash('file added')
            # df = pd.read_csv(form.csv_file.data)
            # return render_template('upload.html', shape=csv_data, form=form)
        return render_template('upload.html', title='Upload', form=form)

    @app.route('/reload')
    def reload():
        global to_reload
        to_reload = True
        return redirect('/dashboard')

    if to_reload:
        with app.app_context():
            from application.plotlydash.Dashboards import IrisDashboard
            app = IrisDashboard(app).create_dashboard(session['dashboard_config'])

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

