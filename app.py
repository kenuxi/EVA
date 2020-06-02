from application import create_app

app = create_app()
app.config['SECRET_KEY'] = 'b042acdb071518100e25f40b93088487'

if __name__ == "__main__":
    app.run(debug=True)