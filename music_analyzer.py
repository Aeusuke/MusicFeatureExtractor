import essentia.standard as es
import constants as cons
import numpy as np
from features import low_level_features, mood_features
import os


class MusicAnalyzer:
    def __init__(self, uploaded_music_name):
        self.__uploaded_music_name = uploaded_music_name
        self.__start_time = 0
        self.__end_time = 0
        self.__Power_Storage = None
        self.__Scaling_Storage = None
        self.__Coefficient_Storage = None
        self.__Max_Storage = None
        self.__Min_Storage = None
        self.__X_Storage = None
        self.__Y_Storage = None
        self.__weights = None
        self.__Data_Storage = None
        self.__eval_steps = 0
        self.__final_y = None
        self.__final_mood_array = None

    def validate_start_end_time(self, start_time_text, end_time_text):
        mono_audio_file = es.EasyLoader(filename=self.__uploaded_music_name)()
        music_duration = es.Duration()(mono_audio_file)
        temp_start_time = 0.0
        temp_end_time = music_duration
        try:
            if start_time_text != "":
                temp_start_time = float(start_time_text)

            if end_time_text != "":
                temp_end_time = float(end_time_text)
        except ValueError:
            return False, "Start time or end time must be a floating point number"
        if temp_start_time >= temp_end_time:
            return False, "Start time must be less than end time"
        if temp_start_time > music_duration or temp_end_time > music_duration:
            return False, "Start time or end time must not be greater than the duration"
        if temp_start_time < 0 or temp_end_time < 0:
            return False, "Start time or end time must be non-negative"
        self.__start_time = temp_start_time
        self.__end_time = temp_end_time
        return True, ""

    def get_audio_features(self):
        self.__get_text_data()
        self.__adjust_data(self.__X_Storage)
        self.__train_data()
        self.__split_music()
        self.__extract_music()
        average_data_storage = self.__compute_average_of_features(self.__Data_Storage)
        average_data_storage = average_data_storage.reshape(1, cons.NUMBER_OF_FEATURES)
        self.__adjust_data(average_data_storage)
        self.__calculate_high_level_features(average_data_storage)
        self.__get_primary_secondary_mood()
        return np.ndarray.tolist(self.__final_y), self.__final_mood_array

    def __get_primary_secondary_mood(self):
        mood_final_y = self.__final_y[2:10]
        max_index = -1
        second_max_index = -1
        max_val = -np.inf
        second_max_val = -np.inf
        for i in range(len(mood_final_y)):
            val = mood_final_y[i]
            if val > max_val:
                max_val = val
                max_index = i
        for i in range(len(mood_final_y)):
            if i == max_index:
                continue
            val = mood_final_y[i]
            if val > second_max_val:
                second_max_val = val
                second_max_index = i
        primary_mood = mood_features[max_index]
        secondary_mood = mood_features[second_max_index]
        self.__final_mood_array = [primary_mood, secondary_mood]

    def __calculate_high_level_features(self, average_data_storage):
        self.__final_y = np.append(1, average_data_storage) @ self.__weights

    def __compute_average_of_features(self, data_storage):
        s, n = data_storage.shape
        average_data_storage = np.zeros(n)
        for i in range(n):
            total_sum = 0.0
            for j in range(s):
                total_sum += data_storage[j, i]
            average_data_storage[i] = total_sum / s
        return average_data_storage

    def __extract_music(self):
        self.__Data_Storage = np.zeros([self.__eval_steps, cons.NUMBER_OF_FEATURES])
        for step in range(self.__eval_steps):
            features, features_frames = es.MusicExtractor(lowlevelStats=["mean", "stdev", "dmean"],
                                                          rhythmStats=["mean", "stdev", "dmean"],
                                                          tonalStats=["mean", "stdev", "dmean"],
                                                          lowlevelSilentFrames="keep",
                                                          tonalSilentFrames="keep",
                                                          mfccStats=["mean"],
                                                          gfccStats=["mean"])(f"./Temp/temp{step}.mp3")
            index = 0
            for i in low_level_features:
                feature = features[i]
                if type(feature) is np.ndarray:
                    if len(feature) > cons.MAX_FEATURE_ARRAY_LENGTH:
                        feature = feature.mean()
                        self.__Data_Storage[step, index] = feature
                        index += 1
                    else:
                        for j in range(len(feature)):
                            self.__Data_Storage[step, index] = feature[j]
                            index += 1
                else:
                    if feature == "major":
                        feature = 1
                    elif feature == "minor":
                        feature = 0
                    self.__Data_Storage[step, index] = feature
                    index += 1

    def __split_music(self):
        music_duration = self.__end_time - self.__start_time
        self.__eval_steps = int(music_duration / cons.EVALUATION_LENGTH) + 1
        audio_file, a, b, c, d, e = es.AudioLoader(filename=self.__uploaded_music_name)()
        if not os.path.exists("./Temp"):
            os.mkdir("./Temp")
        for j in range(self.__eval_steps):
            temp_audio_file = es.StereoTrimmer(startTime=self.__start_time + music_duration / self.__eval_steps * j,
                                               endTime=self.__start_time + music_duration /
                                                       self.__eval_steps * (j + 1))(audio_file)
            es.AudioWriter(filename="./Temp/temp" + str(j) + ".mp3", format="mp3")(temp_audio_file)

    def __train_data(self):
        n, s = self.__X_Storage.shape
        ones_array = np.ones([n, 1])
        lamb = cons.REGRESSION_REGULARIZATION_LAMBDA
        x = np.append(ones_array, self.__X_Storage, axis=1)
        y = self.__Y_Storage
        self.__weights = np.linalg.inv(np.transpose(x) @ x + lamb * np.identity(s + 1)) @ (np.transpose(x) @ y)

    def __adjust_data(self, data_storage):
        n, num_of_features = data_storage.shape
        for i in range(n):
            for j in range(num_of_features):
                raw_num = float(data_storage[i, j])
                adjusted_num = 5.0
                max_num = float(self.__Max_Storage[j])
                min_num = float(self.__Min_Storage[j])
                pow_num = float(self.__Power_Storage[j])
                scaling_num = float(self.__Scaling_Storage[j])
                coefficient_num = float(self.__Coefficient_Storage[j])
                if not np.isnan(raw_num):
                    if raw_num > max_num:
                        adjusted_num = 10.0
                    elif raw_num < min_num:
                        adjusted_num = 0.0
                    else:
                        adjusted_num = coefficient_num * (((raw_num - min_num) * scaling_num) ** pow_num)
                data_storage[i, j] = adjusted_num

    def __get_text_data(self):
        with open("Data/Power.txt", "r") as power_opener:
            for line in power_opener:
                self.__Power_Storage = line.split()
        with open("Data/Scaling.txt", "r") as scaling_opener:
            for line in scaling_opener:
                self.__Scaling_Storage = line.split()
        with open("Data/Coefficient.txt", "r") as coefficient_opener:
            for line in coefficient_opener:
                self.__Coefficient_Storage = line.split()
        with open("Data/Max.txt", "r") as max_opener:
            for line in max_opener:
                self.__Max_Storage = line.split()
        with open("Data/Min.txt", "r") as min_opener:
            for line in min_opener:
                self.__Min_Storage = line.split()
        with open("Data/training_X.txt", "r") as training_x_opener:
            temp_x_storage = training_x_opener.read().splitlines()
        with open("Data/training_Y.txt", "r") as training_y_opener:
            temp_y_storage = training_y_opener.read().splitlines()
        total_songs = len(temp_y_storage)
        for m in range(total_songs):
            temp_x_storage[m] = temp_x_storage[m].split()
            temp_y_storage[m] = temp_y_storage[m].split()
        self.__X_Storage = np.asfarray(temp_x_storage)
        self.__Y_Storage = np.asfarray(temp_y_storage)
