import os
from flask import Flask, request, redirect, url_for, send_from_directory,render_template
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
import pandas as pd
import datetime as dt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
UPLOAD_FOLDER = 'C:/Users/balar/OneDrive/Desktop/Intern/uploads'
CROP_FOLDER = 'C:/Users/balar/OneDrive/Desktop/Intern/crop'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROP_FOLDER'] = CROP_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/options', methods=['GET', 'POST'])
def options():
    return render_template('options.html')

@app.route('/options2', methods=['GET', 'POST'])
def options2():
    return render_template('options2.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

@app.route('/owneroptions', methods=['GET', 'POST'])
def owneroptions():
    return render_template('owneroptions.html')

@app.route('/add', methods=['GET', 'POST'])
def add():
    return render_template('add.html')

@app.route('/service', methods=['GET', 'POST'])
def service():
    return render_template('form.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    return render_template('search.html')

@app.route('/hi', methods=['GET', 'POST'])
def hi():
    return render_template('hi.html')

@app.route('/fetch', methods=['GET', 'POST'])
def fetch():
    return render_template('fetch.html')

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    return render_template('hello.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv('data/servicedata.csv')

    # Convert bought_date column to datetime format
    data['bought_date'] = pd.to_datetime(data['bought_date'])

    # Calculate days since vehicle was bought
    data['days_since_bought'] = (pd.Timestamp.now().normalize() - data['bought_date']).dt.days

    # Split data into features (X) and target (y)
    X = data.drop(['service_date', 'bought_date'], axis=1)
    y = data['service_date']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model pipeline
    model = make_pipeline(
        StandardScaler(),
        SVR(C=1.0, epsilon=0.2, kernel='rbf')
    )

    # Train the model
    model.fit(X_train, y_train)
    # Get user input from form
    kilometres = int(request.form['kilometres'])
    engine_problem = int(request.form['engine_problem'])
    battery_problem = int(request.form['battery_problem'])
    breaking_issues = int(request.form['breaking_issues'])
    bought_date = request.form['bought_date']
    
    # Convert bought_date to datetime format
    bought_date = pd.to_datetime(bought_date)

    # Calculate days since vehicle was bought
    days_since_bought = (pd.Timestamp(dt.date.today()) - bought_date).days
    # Load dataset

    # Make prediction with model
    input_data = [[kilometres, engine_problem, battery_problem, breaking_issues, days_since_bought]]
    service_date_num = model.predict(input_data)[0]
    service_date = dt.date.today() + dt.timedelta(days=service_date_num)
    
    # Render results template with predicted service date
    return render_template('results.html', service_date=service_date)

if __name__ == '__main__':
    app.run()