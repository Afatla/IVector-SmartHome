from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
import wave2ivec
import json
import numpy as np

app = Flask(__name__)

# Read the swagger.yml file to configure the endpoints
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "REST API"
    }
)

app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
iv = wave2ivec.IVector()

# get ivector from wav
@app.route('/ivector', methods=['POST'])
def get_ivector():
    print(request.files)
    file = request.files['file']
    data = iv.process_wav(file, mode="ivector")
    return jsonify({'ivector': data.tolist()})

# get score out of 2 groups of ivector
@app.route('/score', methods=['POST'])
def count_score():
    file = request.files['file']
    score = iv.get_score_from_ivectors_zip(file)
    return jsonify({"Score from ivectors" : score.tolist()})

# get some matrices, TV, LDA, PLDA
@app.route('/matrices', methods=['GET'])
def get_ready_matrices():
    json = request.get_json(force=True)
    type = json['type']
    if type.lower() == u"tv":
        return jsonify({"TV matrices" : iv.get_matrices("TV").tolist()})
    elif type.lower() == u"lda":
        return jsonify({"LDA matrices": iv.get_matrices("LDA").tolist()})
    elif type.lover() == u"plda_c":
        return jsonify({"PLDA c matrices": iv.get_matrices("PLDA_c").tolist()})
    elif type.lover() == u"plda_gamma":
        return jsonify({"PLDA gamma matrices": iv.get_matrices("PLDA_gamma").tolist()})
    elif type.lover() == u"plda_k":
        return jsonify({"PLDA k matrices": iv.get_matrices("PLDA_k").tolist()})
    elif type.lover() == u"plda_lambda":
        return jsonify({"PLDA lambda matrices": iv.get_matrices("PLDA_lambda").tolist()})
    else:
        return jsonify({'message' : "try again"})

# get mfcc out of wav
@app.route('/mfcc', methods=['POST'])
def get_mfcc():
    file = request.files['file']
    data = iv.process_wav(file, mode="mfcc")
    return jsonify({'mfcc': data.tolist()})

@app.route('/mfcc2ivector', methods=['POST'])
def mfcc_to_ivector():
    file = request.files['file']
    mfcc_array = json.load(file)['Mfcc']
    mfcc_array = np.asarray(mfcc_array)
    data = iv.mfcc_to_ivector(mfcc_array)
    return jsonify({'ivector': data.tolist()})

# get statistics out of wav
@app.route('/stats', methods=['POST'])
def get_stats():
    type = request.form['type']
    file = request.files['f']
    if type.lower() == u"f":
        return jsonify({'Statistics first order' : iv.process_wav(file, mode="statistics")[0].tolist()})
    elif type.lower() == u"n":

        return jsonify({'Statistics zero order': iv.process_wav(file, mode="statistics")[1].tolist()})
    else:
        return jsonify({'message': "try again"})

if __name__ == '__main__':
    print "test"
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)



