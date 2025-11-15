from flask import Flask, request, jsonify
from flask_cors import CORS
#imports the model itself
from predictor import Model

#create flask app
app = Flask(__name__)

#connects future frontend
CORS(app)


print("starting server")

#creates instance of model class
predictor = Model()



#whenever get does /, run func below
@app.route('/')
def home():
    
    #returns json object to confirm running
    #health status
    return jsonify({
        'status': 'running',
        'message': 'backend active',
        #lists out workig categories
        'available_categories': list(predictor.models.keys())
    })

#actual prediction function
@app.route('/predict', methods=['POST'])
def predict():
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image provided'
        }), 400 
    
    # check if category was sent
    if 'category' not in request.form:
        return jsonify({
            'success': False,
            'error': 'No category provided'
        }), 400
    
    
    image_file = request.files['image']
    category = request.form['category']
    
    # read image as bytes
    image_bytes = image_file.read()
    
    # uses predict function in predict.py
    result = predictor.predict(image_bytes, category)
    
    # return
    if result['success']:
        return jsonify(result), 200  
    else:
        return jsonify(result), 400  #

# run
if __name__ == '__main__':
    print("------------------------------")
    print("starting....")
    print("------------------------------")
    print(f"categories : {list(predictor.models.keys())}")
    print("\nhttp://localhost:5000")
    print("------------------------------")
    
    # start
    app.run(
        debug=True,      # Show errors in terminal
        host='0.0.0.0',  # Allow external connections
        port=5000        # Run on port 3000
    )