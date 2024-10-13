import openai
import speech_recognition as sr
import logging
from elevenlabs import generate, stream

# Set up logging for better debugging and visibility
logging.basicConfig(level=logging.INFO)


class VoiceAssistant:
    def __init__(self):
        # Include your API keys here
        self.elevenlabs_api_key = ""
        self.openai_api_key = ""
        openai.api_key = self.openai_api_key

        self.full_conversation = [
            {"role": "system", "content": "You are Beacon, an AI voice assistant designed to help with emotional support and general tasks."}
        ]

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            logging.info("Listening...")
            audio = self.recognizer.listen(source)
        try:
            transcript = self.recognizer.recognize_google(audio)
            logging.info(f"User said: {transcript}")
            return transcript
        except sr.UnknownValueError:
            logging.info("Speech not recognized, please try again.")
            return None
        except sr.RequestError as e:
            logging.error(f"Error with the speech recognition service: {e}")
            return None

    def handle_transcription(self, user_input):
        # Add the user's transcription to the conversation
        self.full_conversation.append({"role": "user", "content": user_input})
        logging.info(f"User said: {user_input}")

        # Get response from OpenAI ChatCompletion API
        try:
            logging.info("Sending request to OpenAI...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
                messages=self.full_conversation
            )
            ai_response = response.choices[0].message['content']
            self.full_conversation.append(
                {"role": "assistant", "content": ai_response})
            logging.info(f"Beacon replied: {ai_response}")

            # Convert the AI response to speech
            self.text_to_speech(ai_response)

        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            self.text_to_speech("I'm sorry, I couldn't process that request.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            self.text_to_speech("An unexpected error occurred.")

    def text_to_speech(self, text):
        logging.info(f"Converting to speech: {text}")
        try:
            audio_stream = generate(
                api_key=self.elevenlabs_api_key,
                text=text,
                voice="Jessica",
                stream=True
            )
            stream(audio_stream)
        except Exception as e:
            logging.error(f"Error during text-to-speech conversion: {e}")
            print("Failed to convert text to speech.")


def listen_for_wake_word(recognizer, microphone):
    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            logging.info("Listening for the wake word...")
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            logging.info(f"Recognized command: {command}")
            if "beacon" in command.lower():
                logging.info("Wake word detected.")
                return
        except sr.UnknownValueError:
            logging.info("Speech not recognized, please try again.")
        except sr.RequestError as e:
            logging.error(f"Error with the speech recognition service: {e}")


if __name__ == "__main__":
    # Start by greeting the user
    greeting = "Hello! I am Beacon, your personal AI voice assistant. How can I help you today?"
    beacon_ai = VoiceAssistant()
    beacon_ai.text_to_speech(greeting)

    # Listen for the wake word to activate the assistant
    while True:
        listen_for_wake_word(beacon_ai.recognizer, beacon_ai.microphone)
        logging.info("Wake word acknowledged. Listening for user input...")
        user_input = beacon_ai.listen()
        if user_input:
            beacon_ai.handle_transcription(user_input)
