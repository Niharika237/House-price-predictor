from flask import Flask, request, render_template, jsonify
import pandas as pd 
import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model("hyd_house_xgb_revised.json")

app = Flask(__name__)
    
@app.route('/')
def home():
    return render_template('home.html')  

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            location = float(request.form['location'])
            size = int(request.form['size']) 
            bath = int(request.form['bathroom'])
            balcony = int(request.form['balcony'])
            total_sqft = float(request.form['total_sqft'])

            input_df = pd.DataFrame([[location, size, balcony, bath, total_sqft]],
                                    columns=['location', 'size(bedrooms)', 'balcony', 'bathroom', 'total_sqft'])

            prediction = model.predict(input_df)[0]
            prediction *= 0.5  

            return render_template('show.html', inf=round(prediction, 2))

        except KeyError as e:
            return f"Missing or invalid input: {e}", 400
        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
    