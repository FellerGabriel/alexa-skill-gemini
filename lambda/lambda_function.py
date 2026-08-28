# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import logging
import re
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
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is not set. Create a .env file next to this one with GOOGLE_API_KEY=<key>.")
# Free-tier models. Pro is not available on the free tier, it returns 429 with limit 0.
# gemini-flash-latest answers better but has been returning 503 under load, so the lite model leads
# and the better one is the fallback. Swap them with the env vars when the load eases.
MODEL = os.getenv('GEMINI_MODEL', 'gemini-flash-lite-latest')
FALLBACK_MODEL = os.getenv('GEMINI_FALLBACK_MODEL', 'gemini-flash-latest')
# API endpoint URL
URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}"
# Headers for the request
headers = {
    'Content-Type': 'application/json',
}
# Alexa drops the response after about 8 seconds, so keep the thinking shallow and the timeouts tight.
# maxOutputTokens also pays for the thinking tokens, so a tight budget truncates the answer mid-sentence.
GENERATION_CONFIG = {
    "thinkingConfig": {"thinkingLevel": "low"},
    "maxOutputTokens": 800,
}
PRIMARY_TIMEOUT_SECONDS = 5
FALLBACK_TIMEOUT_SECONDS = 3
# While set, the skill says why a call failed out loud, so a failure can be diagnosed from the
# simulator without opening CloudWatch. Set GEMINI_DEBUG=0 in .env once the skill works.
DEBUG_ERRORS = os.getenv('GEMINI_DEBUG', '1') != '0'
# Reason of the last failed call, spoken when DEBUG_ERRORS is on
last_error = None
# Conversation history sent to Gemini, rebuilt on every skill launch
data = {
    "contents": []
}

# Speech and prompts per language, keyed by the language part of the request locale
LANGUAGE_STRINGS = {
    "pt": {
        "system_prompt": "Você é o Gemini, do Google, respondendo por voz através de uma skill da Alexa. Você não é a Alexa: se perguntarem quem você é, diga que é o Gemini. Responda sempre em português do Brasil, em texto puro para ser falado em voz alta: sem markdown, sem listas numeradas, sem asteriscos e sem emojis. Seja claro e breve, no máximo três frases, a menos que peçam mais detalhes. Você só conversa: não controla dispositivos, luzes, televisão nem casa inteligente, e não cria alarmes, timers, lembretes, listas ou compras. Se pedirem uma dessas coisas, nunca diga que fez ou que vai fazer: diga que não consegue e oriente a pedir direto à Alexa.",
        "greeting": "Olá, eu sou seu assistente com o Gemini. Como posso ajudar?",
        "ack": "Combinado!",
        "reprompt": "Mais alguma pergunta?",
        "no_answer": "Não recebi resposta para o seu pedido",
        "not_understood": "Não entendi a sua pergunta. Pode repetir?",
        "help": "Você pode me perguntar qualquer coisa e eu respondo com a ajuda do Gemini. O que você quer saber?",
        "goodbye": "Até logo!",
        "error": "Desculpe, tive um problema ao fazer o que você pediu. Tente novamente.",
    },
    "en": {
        "system_prompt": "You are Gemini, by Google, answering by voice through an Alexa skill. You are not Alexa: if asked who you are, say you are Gemini. Always answer in English, in plain text meant to be spoken out loud: no markdown, no numbered lists, no asterisks and no emojis. Be clear and brief, at most three sentences, unless more detail is requested. You only talk: you do not control devices, lights, televisions or smart home, and you do not create alarms, timers, reminders, lists or purchases. If asked for any of those, never say you did it or will do it: say you cannot and tell the user to ask Alexa directly.",
        "greeting": "Hello, I'm your Gemini Chat Bot. How can I help you?",
        "ack": "Understood!",
        "reprompt": "Any other questions?",
        "no_answer": "I did not receive a response to your request",
        "not_understood": "I did not catch your question. Could you repeat it?",
        "help": "You can ask me anything and I will answer with the help of Gemini. What would you like to know?",
        "goodbye": "Goodbye!",
        "error": "Sorry, I had trouble doing what you asked. Please try again.",
    },
}


def get_strings(handler_input):
    """Return the speech strings matching the locale of the current request."""
    locale = handler_input.request_envelope.request.locale or "en-US"
    return LANGUAGE_STRINGS.get(locale.split("-")[0], LANGUAGE_STRINGS["en"])


def for_speech(text):
    """Strip markdown markers, which Alexa would otherwise read out loud."""
    text = re.sub(r'[*_`#]+', '', text)
    return re.sub(r'\s+\n', '\n', text).strip()


def call_model(model, timeout):
    """POST the current history to one model. Returns the answer text, or None on failure."""
    global last_error
    if not GOOGLE_API_KEY:
        last_error = "a chave da API não foi encontrada no arquivo .env"
        logger.error("GOOGLE_API_KEY is empty, not calling %s", model)
        return None

    payload = dict(data, generationConfig=GENERATION_CONFIG)
    try:
        response = requests.post(
            URL_TEMPLATE.format(model, GOOGLE_API_KEY),
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        last_error = "o modelo {} demorou mais de {} segundos".format(model, timeout)
        logger.error("Request to %s timed out after %ss", model, timeout)
        return None
    except requests.exceptions.RequestException as error:
        last_error = "não consegui conectar na API do Gemini"
        logger.error("Request to %s failed: %s", model, error)
        return None

    if response.status_code != 200:
        last_error = "o modelo {} respondeu erro {}".format(model, response.status_code)
        logger.error("Model %s returned %s: %s", model, response.status_code, response.text[:500])
        return None

    body = response.json()
    # Identifies which model actually served the answer, straight from the API response
    logger.info("Answered by %s, responseId %s, tokens %s",
                body.get("modelVersion"),
                body.get("responseId"),
                body.get("usageMetadata", {}).get("totalTokenCount"))
    candidate = body.get("candidates", [{}])[0]
    finish_reason = candidate.get("finishReason")
    if finish_reason not in (None, "STOP"):
        logger.warning("Model %s stopped with finishReason %s", model, finish_reason)
    return (candidate.get("content", {})
        .get("parts", [{}])[0]
        .get("text"))


def ask_gemini(text):
    """Append the user turn, call Gemini and append the model turn. Returns the answer or None."""
    data["contents"].append({
        "role": "user",
        "parts": [{
            "text": text
        }]
    })
    answer = call_model(MODEL, PRIMARY_TIMEOUT_SECONDS)
    if answer is None:
        answer = call_model(FALLBACK_MODEL, FALLBACK_TIMEOUT_SECONDS)
    if answer is None:
        # Drop the unanswered turn so the next question is not sent with a dangling user message
        data["contents"].pop()
        return None

    data["contents"].append({
        "role": "model",
        "parts": [{
            "text": answer
        }]
    })
    return for_speech(answer)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        strings = get_strings(handler_input)
        # Start a fresh conversation, seeded with the language instructions. The greeting is local:
        # calling Gemini just to say hello made the launch fail whenever the API was unavailable.
        data["contents"] = [
            {"role": "user", "parts": [{"text": strings["system_prompt"]}]},
            {"role": "model", "parts": [{"text": strings["ack"]}]},
        ]
        speak_output = strings["greeting"]

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
        if not query:
            logger.warning("ChatIntent matched with an empty query slot")
            return (
                handler_input.response_builder
                    .speak(strings["not_understood"])
                    .ask(strings["reprompt"])
                    .response
            )

        text = ask_gemini(query)
        if text is not None:
            speak_output = text
        elif DEBUG_ERRORS and last_error:
            speak_output = "Falhou porque " + last_error
        else:
            speak_output = strings["no_answer"]

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
