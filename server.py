from feature_extraction.run_extraction import feature_extract_segments
from preprocess.preprocess import preprocess_audio
import sys
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from pydub import AudioSegment
from pathlib import Path
from http import HTTPStatus
from enum import Enum
from dataclasses import dataclass
import pickle
import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

sys.path.append("preprocess/")
sys.path.append("feature_extraction/")
>>>>>> > 0fc94b9(update: server.py)

app = Flask(__name__)
uploads_path = "tmp/uploads"


class SD_Class(Enum):
    NSD = 0
    SD = 1


@dataclass
class Classification:
    sd: SD_Class
    confidence_score: float
    result: str
    # other fields here


@app.route("/upload", methods=["POST"])
def Upload():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), HTTPStatus.BAD_REQUEST

    audio_file = request.files["audio"]
    if audio_file.filename:
        Path(uploads_path).mkdir(parents=True, exist_ok=True)

        uploads = Path(uploads_path)

        file_path = uploads / secure_filename(audio_file.filename)
        audio_file.save(file_path)

        wav_file = convertWAV(file_path)
        clf = classify(wav_file)

        return (
            jsonify(
                {
                    "class": clf.sd.value,
                    "result": clf.result,
                }
            ),
            HTTPStatus.OK,
        )

    return (
        jsonify({"error": "There was a problem saving the file"}),
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )


def predict_features(features, svm, pca):
    pre_counter = 0
    post_counter = 0
    avg_confidence_score = 0.0

    for i, feature in enumerate(features):
        print(f"Processing feature {i + 1}")

        # Flatten and normalize the feature array
        feature_flattened = np.asarray(feature).flatten()
        feature_normalized = (
            feature_flattened / np.max(np.abs(feature_flattened))
            if np.max(np.abs(feature_flattened)) != 0
            else feature_flattened
        )

        # Reshape the feature array for PCA transformation
        feature_reshaped = feature_normalized.reshape(1, -1)

        expected_features = pca.components_.shape[1]
        if feature_flattened.shape[0] != expected_features:
            raise ValueError(
                f"Feature mismatch! Expected {expected_features} features, but got {
                    feature_flattened.shape[0]
                }."
            )

        # Apply PCA transformation
        feature_pca = pca.transform(feature_reshaped)

        # predict class label
        y_pred = svm.predict(feature_pca)

        # 0 - post. 1 - pre
        # print(f"bacc: {balanced_accuracy_score(svm.classes_, y_pred)}")
        # print(f"best params of the trained model: {svm.best_params_}")
        # print("All classes in the model:", svm.classes_)
        # print("Predicted class:", y_pred)

        # print(f"post counts: {post_counter}")
        # print(f"pre counts: {pre_counter}")

        if hasattr(svm, "decision_function"):
            decision_scores = svm.decision_function(feature_pca)
            probs = softmax(decision_scores)
        elif hasattr(svm, "predict_proba"):
            probs = svm.predict_proba(feature_pca)
        else:
            raise AttributeError(
                "SVM model does not support decision_function or predict_proba."
            )

        # logits = np.array([1.0, 0.1])
        # probs = softmax(logits)
        confidence = np.max(probs)
        print(f"confidence score: {confidence}")

        # Update counters based on prediction
        if y_pred == svm.classes_[0]:
            post_counter += 1
        elif y_pred == svm.classes_[1]:
            pre_counter += 1

        print(f"Predicted class for feature {i + 1}: {y_pred}")
        print(f"Confidence score: {confidence}")

        avg_confidence_score += confidence

    avg_confidence_score = avg_confidence_score / len(features)
    print(f"Pre (non-sleep-deprived) features counts: {pre_counter}")
    print(f"Post (non-sleep-deprived) features counts: {post_counter}")
    print(f"average CFS: {avg_confidence_score}")

    return pre_counter, post_counter, avg_confidence_score


def classify(audio_path: Path) -> Classification:
    """
    Predict the class labels for the given STM features array of 3D using the trained SVM and PCA models.

    Args:
        features (list): List of feature arrays (e.g., STRF features).
        svm_path (str): Path to the trained SVM model (.pkl file).
        pca_path (str): Path to the trained PCA model (.pkl file).
    """
    svm_path = Path("./population_level_svm.pkl")
    pca_path = Path("./population_level_pca.pkl")
    test_sample_path = Path(
        # "~/Research/Sleep Deprivation Detection using voice/strf_data_new.pkl"
        # "~/Research/Sleep Deprivation Detection using voice/dataset/osf/stmtf/strf_session_post_subjectNb_01_daySession_01_segmentNb_0.pkl"
        # "~/github/16Khz-models/feature_extraction/gian_data_new.pkl"
        "./strf_data_new.pkl"
        # "~/github/16Khz-models/feature_extraction/pkls/segment_72_strf.pkl"
    )

    # Load the SVM and PCA models using pickle
    with open(svm_path, "rb") as f:
        svm = pickle.load(f)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    # Define the output directory, if necessary to be stored
    output_dir_processed = "preprocess/preprocessed_audio/processed_audio/"
    output_dir_features = "feature_extraction/extracted_features/feature"

    segments, sr = preprocess_audio(audio_path, output_dir_processed)

    # Print details
    print(f"Number of segments: {len(segments)}")
    print(f"Sampling rate: {sr} Hz")

    # Feature Extraction
    features = feature_extract_segments(segments, output_dir_features, sr)

    # test_sample = pickle.load(test_sample_path)
    with open(test_sample_path, "rb") as f:
        test_sample = pickle.load(f)

    # print(type(test_sample), test_sample)
    np.set_printoptions(threshold=np.inf)
    #
    # magnitude_strf = np.abs(test_sample)
    #
    # # STRF (128, 8, 22)
    # test_sample = np.mean(magnitude_strf, axis=0)
    # print(test_sample["strf"])

    pre_count, post_count, avg_confidence_score = predict_features(
        features, svm, pca)

    if post_count > pre_count:
        return Classification(
            sd=SD_Class.SD,
            confidence_score=avg_confidence_score,  # 82.6%
            result="You are sleep deprived.",
        )
    else:
        return Classification(
            sd=SD_Class.NSD,
            confidence_score=avg_confidence_score,  # 82.6%
            result="You are not sleep deprived.",
        )


def convertWAV(audio: Path) -> Path:
    wav = audio.with_suffix(".wav")
    file = AudioSegment.from_file(audio)
    file.export(wav, format="wav")

    audio.unlink()
    return Path(wav)
