import os
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

PROJECT_ID = "jouw_project_id"
RECOGNIZER_ID = "jouw_recognizer_id"

def create_recognizer(project_id: str, recognizer_id: str) -> cloud_speech.Recognizer:
    client = SpeechClient()
    request = cloud_speech.CreateRecognizerRequest(
        parent=f"projects/{project_id}/locations/global",
        recognizer_id=recognizer_id,
        recognizer=cloud_speech.Recognizer(
            default_recognition_config=cloud_speech.RecognitionConfig(
                language_codes=["nl-NL"],
                model="long",
            ),
        ),
    )
    operation = client.create_recognizer(request=request)
    recognizer = operation.result()
    return recognizer

def transcribe_audio(project_id: str, recognizer_id: str, audio_path: str) -> cloud_speech.RecognizeResponse:
    client = SpeechClient()
    with open(audio_path, "rb") as f:
        content = f.read()
    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/{recognizer_id}",
        content=content,
    )
    response = client.recognize(request=request)
    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/nillavanilla/Downloads/stt2langchain1-a71fb5485650.json"
    audio_file_path = "path_to_your_audio_file.wav"
    
    # Het aanmaken van de recognizer zou je normaal gesproken maar één keer doen.
    # Daarna zou je het kunnen hergebruiken voor verschillende transcribe taken.
    # Voor dit voorbeeld doen we beide in dezelfde uitvoering.
    create_recognizer(PROJECT_ID, RECOGNIZER_ID)
    transcribe_audio(PROJECT_ID, RECOGNIZER_ID, audio_file_path)
