from flask import Flask, request, jsonify
from flask_cors import CORS
from totalProcess import whisper
from totalProcess import summarizeTexts
from totalProcess import classify_sentences
from totalProcess import getScore
from totalProcess import videoProcess
from totalProcess import getImageScore
from totalProcess import checkConcentration
import tempfile

app = Flask(__name__)
CORS(app)

@app.route('/api/text_summarization', methods=['POST'])
def summarization() :
    file = request.files['file']
    
    transcripts = whisper(file)
    summaries = summarizeTexts(transcripts)
    
    return summaries

@app.route('/api/audio_analysis', methods=['POST'])
def audioAnalysis() :
    file = request.files['file']
    
    transcripts = whisper(file)
    classification_results = classify_sentences(transcripts)

    score = getScore(classification_results)
    
    return jsonify({"score": score})

@app.route('/api/video_analysis', methods=['POST'])
def videoAnalysis() :
    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        video_path = tmp.name
    
    predictions = videoProcess(video_path)
    print(predictions)
    score = getImageScore(predictions)
    
    return jsonify({"score": score})

@app.route('/api/concentration', methods=['POST'])
def concentration() :
    file = request.files['file']
    
    concentrationRatio = checkConcentration(file)
    
    return concentrationRatio


if __name__ == '__main__' :
    app.run(host='0.0.0.0', port=5000, debug=True)