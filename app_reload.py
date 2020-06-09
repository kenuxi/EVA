from werkzeug.serving import run_simple
from flask import Flask, render_template, url_for, redirect, request, session
from forms import HomePageForm
from config import app_secret_key

to_reload = False
app = None


def get_app():
    # this need to be cleaner
    global app
    app = Flask(__name__, instance_relative_config=False,
                template_folder='application/templates')
    app.config['SECRET_KEY'] = app_secret_key

    @app.route('/', methods=['GET', 'POST'])
    def home():
        form = HomePageForm(request.form)
        if request.method == 'POST':
            # Use this place arguments to create dashboards in reload()
            dashboard_config = {
                'data': form.select.data,
                'target': form.target.data,
                'pca': form.pca.data,
                'tsne': form.tsne.data
            }
            session['dashboard_config'] = dashboard_config
            return redirect(url_for('reload'))
        return render_template('home.html', title='Home',
                               form=form)

    @app.route("/about")
    def about():
        return render_template('about.html', title='About')

    @app.route('/reload')
    def reload():
        global to_reload
        global app
        to_reload = True
        with app.app_context():
            from application.plotlydash.Dashboards import IrisDashboard
            app = IrisDashboard(app).create_dashboard(session['dashboard_config'])
        return session['dashboard_config']

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
