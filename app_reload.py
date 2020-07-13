from werkzeug.serving import run_simple
from flask import Flask, render_template, url_for, redirect, flash
from flask_uploads import configure_uploads, UploadSet
from forms import SelectFileForm, UploadForm, VisForm
from config import app_secret_key, session
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
        file_form = SelectFileForm()
        up_form = UploadForm()
        vis_form = VisForm()
        if file_form.file_submit.data and file_form.validate_on_submit():
            df = pd.read_csv(file_form.file.data)
            session['filename'] = file_form.file.data
            session['DF'] = df
            vis_form.target.choices = [(str(col), str(col)) for col in df.columns]
            return render_template('home.html', title='Home',
                                   df=df,
                                   file_form=file_form,
                                   up_form=up_form,
                                   vis_form=vis_form)

        elif up_form.csv_submit.data and up_form.validate_on_submit():
            csv_data = up_form.csv_file.data
            filename = csv_data.filename
            if filename in os.listdir(os.path.join('data')):
                flash('Filename exists!', 'danger')
                return redirect(url_for('home'))
            csv_data.save(os.path.join('data', filename))
            flash('Your file has been Added!', 'success')
            return redirect(url_for('home'))

        elif vis_form.submit.data:
            dashboard_config = {'location': session['filename'],
                                'target': vis_form.target.data,
                                'PCA': [],
                                'LLE': [],
                                'TSNE': [],
                                'UMAP': [],
                                'ISOMAP': [],
                                'KMAP': []
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

