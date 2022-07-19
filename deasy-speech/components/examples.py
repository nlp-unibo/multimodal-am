from deasy_learning_generic.examples import Example


class ArgAAAIExample(Example):

    def __init__(self, speech_text=None, speech_mfccs=None, speech_audio_data=None, label=None):
        super(ArgAAAIExample, self).__init__(label=label)
        self.speech_text = speech_text
        self.speech_mfccs = speech_mfccs
        self.speech_audio_data = speech_audio_data

        assert speech_text is not None or speech_mfccs is not None or speech_audio_data is not None, \
            f"At least one input should be given." \
            f"Got speech_text={self.speech_text} " \
            f"and speech_mfccs={self.speech_mfccs}."

    def get_data(self):
        return self.speech_text


class MArgExample(Example):

    def __init__(self, text_a=None, text_b=None, audio_a=None, audio_a_data=None,
                 audio_b=None, audio_b_data=None, label=None):
        super(MArgExample, self).__init__(label=label)
        self.text_a = text_a
        self.text_b = text_b
        self.audio_a = audio_a
        self.audio_b = audio_b
        self.audio_a_data = audio_a_data
        self.audio_b_data = audio_b_data

        assert text_a is not None or audio_a is not None or audio_a_data is not None, f"At least one input should be given." \
                                                          f"Got text_a={self.text_a} " \
                                                          f"and audio_a={self.audio_a}"

    def get_data(self):
        if self.text_a is not None:
            return self.text_a + ' ' + self.text_b
        else:
            return ''


class UsElecExample(Example):

    def __init__(self, speech_text=None, speech_mfccs=None, speech_audio_data=None, label=None):
        super(UsElecExample, self).__init__(label=label)
        self.speech_text = speech_text
        self.speech_mfccs = speech_mfccs
        self.speech_audio_data = speech_audio_data

        assert speech_text is not None or speech_mfccs is not None or speech_audio_data is not None, \
            f"At least one input should be given." \
            f"Got speech_text={self.speech_text} " \
            f"and speech_mfccs={self.speech_mfccs}."

    def get_data(self):
        return self.speech_text