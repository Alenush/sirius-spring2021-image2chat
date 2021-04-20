from google_trans_new import google_translator


translator = google_translator()
translate_text = translator.translate('Привет друг',lang_tgt='en')

def translate_me(text, dest="en"):
    """
    Translate via Google
    dest='en' (default) if you want translation ru->en
    dest='ru' if you want en->ru
    """
    translator = google_translator()
    translate_text = translator.translate(text, lang_tgt=dest)

    return translate_text


def translate_text(target, text):

    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print(u"Text: {}".format(result["input"]))
    print(u"Translation: {}".format(result["translatedText"]))
    print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
