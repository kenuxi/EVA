from flask import Flask


def create_app(debug=True):
    app = Flask(__name__, instance_relative_config=False,)

    with app.app_context():
        from application import routes

        from application.plotlydash.Dashboards import IrisDashboard
        app = IrisDashboard(app).create_dashboard()
        return app
