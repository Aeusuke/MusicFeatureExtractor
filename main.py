from music_analyzer import MusicAnalyzer
import os
from scipy.stats import norm
from features import high_level_features
from flask import Flask, render_template, request, redirect, url_for
import constants as cons


app = Flask(__name__)
errorText = " "
final_y_array = []
final_mood_array = []


@app.route('/')
def main_page():
    return render_template('home.html', ErrorText=errorText)


@app.route('/upload_test', methods=['POST'])
def upload_music_test():
    global errorText
    file_reference = request.files['file']
    start_time = request.form['start-time']
    end_time = request.form['end-time']
    file_name = file_reference.filename
    file_extension = os.path.splitext(file_name)[1]
    if not validate_file_extension(file_extension) or file_name == "":
        errorText = "File type must be either .mp3, .wav, or .ogg"
        return redirect(url_for('main_page'))
    uploaded_music_name = f"./Music{file_extension}"
    file_reference.save(uploaded_music_name)
    ma_object = MusicAnalyzer(uploaded_music_name)
    is_valid_duration, errorText = ma_object.validate_start_end_time(start_time, end_time)
    if not is_valid_duration:
        return redirect(url_for('main_page'))
    global final_y_array, final_mood_array
    final_y_array, final_mood_array = ma_object.get_audio_features()

    return redirect(url_for('music_analysis_page', filename=file_name), code=307)


@app.route('/analysis/<filename>', methods=['POST'])
def music_analysis_page(filename):
    percentile_y_array = []
    for y in final_y_array:
        percentile = norm.cdf(y - 5.0)
        percentile_y_array.append(percentile)
    return render_template('analysis.html', finalY=final_y_array, finalMoodArray=final_mood_array,
                           percentileArray=percentile_y_array, highfeatureNames=high_level_features, fileName=filename)


def validate_file_extension(file_extension):
    for ext in cons.SUPPORTED_FILE_TYPES:
        if ext == file_extension:
            return True
    return False






if __name__ == "__main__":
    app.run(debug=True, port=5000)
