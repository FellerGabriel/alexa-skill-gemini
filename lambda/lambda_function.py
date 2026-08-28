# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import logging
import ask_sdk_core.utils as ask_utils
import requests
import json
import os
from dotenv import load_dotenv
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# API endpoint URL
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={}".format(GOOGLE_API_KEY)
# Headers for the request
headers = {
    'Content-Type': 'application/json',
}
# Conversation history sent to Gemini, rebuilt on every skill launch
data = {
    "contents": []
}

# Speech and prompts per language, keyed by the language part of the request locale
LANGUAGE_STRINGS = {
    "pt": {
        "system_prompt": "Olá! Responda sempre em português do Brasil, de forma clara e sem ser prolixo. Combinado?",
        "greeting": "Olá, eu sou seu assistente com o Gemini. ",
        "greeting_suffix": " Como posso ajudar?",
        "reprompt": "Mais alguma pergunta?",
        "text_not_found": "Não encontrei uma resposta",
        "request_error": "Erro na requisição",
        "no_answer": "Não recebi resposta para o seu pedido",
        "help": "Você pode me perguntar qualquer coisa e eu respondo com a ajuda do Gemini. O que você quer saber?",
        "goodbye": "Até logo!",
        "error": "Desculpe, tive um problema ao fazer o que você pediu. Tente novamente.",
    },
    "en": {
        "system_prompt": "Hello! Respond in English clearly and do not be verbose. OK?",
        "greeting": "Hello, I'm your Gemini Chat Bot. ",
        "greeting_suffix": " How can I help you?",
        "reprompt": "Any other questions?",
        "text_not_found": "Text not found",
        "request_error": "Request error",
        "no_answer": "I did not receive a response to your request",
        "help": "You can ask me anything and I will answer with the help of Gemini. What would you like to know?",
        "goodbye": "Goodbye!",
        "error": "Sorry, I had trouble doing what you asked. Please try again.",
    },
}


def get_strings(handler_input):
    """Return the speech strings matching the locale of the current request."""
    locale = handler_input.request_envelope.request.locale or "en-US"
    return LANGUAGE_STRINGS.get(locale.split("-")[0], LANGUAGE_STRINGS["en"])


def ask_gemini(text, strings):
    """Append the user turn, call Gemini and append the model turn. Returns the answer or None."""
    data["contents"].append({
        "role": "user",
        "parts": [{
            "text": text
        }]
    })
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        logger.error("Gemini request failed: %s %s", response.status_code, response.text)
        return None

    response_data = response.json()
    answer = (response_data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", strings["text_not_found"]))
    data["contents"].append({
        "role": "model",
        "parts": [{
            "text": answer
        }]
    })
    return answer


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        strings = get_strings(handler_input)
        # Start a fresh conversation so the language prompt of a previous session is not reused
        data["contents"] = []
        text = ask_gemini(strings["system_prompt"], strings)
        if text is not None:
            speak_output = strings["greeting"] + text + strings["greeting_suffix"]
        else:
            speak_output = strings["request_error"]

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class ChatIntentHandler(AbstractRequestHandler):
    """Handler for Chat Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("ChatIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        strings = get_strings(handler_input)
        query = handler_input.request_envelope.request.intent.slots["query"].value
        text = ask_gemini(query, strings)
        speak_output = text if text is not None else strings["no_answer"]

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(strings["reprompt"])
                .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        strings = get_strings(handler_input)

        return (
            handler_input.response_builder
                .speak(strings["help"])
                .ask(strings["reprompt"])
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        strings = get_strings(handler_input)
        speak_output = strings["goodbye"]

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput) -> Response
        logger.error(exception, exc_info=True)

        speak_output = get_strings(handler_input)["error"]

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ChatIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
