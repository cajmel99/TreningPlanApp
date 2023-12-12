from flask import Flask, render_template, redirect, flash
from upload import upload_blueprint
from predict import predict_blueprint

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace 'your_secret_key' with a secure secret key

app.register_blueprint(upload_blueprint)
app.register_blueprint(predict_blueprint)

# Definicja trasy dla strony głównej
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5004)
