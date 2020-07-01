from werkzeug.serving import run_simple
from flask import Flask, render_template, url_for, redirect, request, flash
from flask_uploads import configure_uploads, UploadSet
from forms import SelectFileForm, AlgorithmForm, UploadForm
from config import app_secret_key, session
import pandas as pd
import os


to_reload = False


def get_app():
    app = Flask(__name__, instance_relative_config=False,
                template_folder='application/templates')
    app.config['SECRET_KEY'] = app_secret_key
    app.config['session'] = {}
    csv_files = UploadSet('data', ('csv',), default_dest=lambda x: 'data')
    configure_uploads(app, csv_files)

    @app.route('/', methods=['GET', 'POST'])
    @app.route('/home', methods=['GET', 'POST'])
    def home():
        file_form = SelectFileForm()
        alg_form = AlgorithmForm()
        up_form = UploadForm()

        if file_form.file_submit.data and file_form.validate_on_submit():
            df = pd.read_csv(file_form.file.data)
            # app.config['session']['df'] = df
            session['file'] = file_form.file.data
            session['DF'] = df
            return render_template('home.html', title='Home',
                                   df=df,
                                   file_form=file_form,
                                   up_form=up_form,
                                   alg_form=alg_form)

        elif up_form.csv_submit.data and up_form.validate_on_submit():
            csv_data = up_form.csv_file.data
            filename = csv_data.filename
            if filename in os.listdir(os.path.join('data')):
                flash('Filename exists!', 'danger')
                return redirect(url_for('home'))
            csv_data.save(os.path.join('data', filename))
            flash('Your file has been Added!', 'success')
            return redirect(url_for('home'))

        elif alg_form.submit.data and alg_form.validate_on_submit():
            session['target'] = alg_form.target.data
            session['algorithm'] = alg_form.algorithm.data
            return str(session['file']) + '<br/>' + str(session['target']) + '<br/>' + str(session['algorithm'])

        return render_template('home.html', title='Home', file_form=file_form, up_form=up_form)

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        form = UploadForm()
        if form.validate_on_submit():
            csv_data = form.csv_file.data
            filename = csv_data.filename
            csv_data.save(os.path.join(app.instance_path, 'photos', filename))
            flash('Your file has been Added!', 'success')
            return redirect(url_for('home'))
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

