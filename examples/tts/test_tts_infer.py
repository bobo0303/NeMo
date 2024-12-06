# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is used as a CI test and shows how to chain TTS and ASR models
"""

from argparse import ArgumentParser
from math import ceil

import librosa
import soundfile
import torch
import time
import csv
import os

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from nemo.utils import logging

# LIST_OF_TEST_STRINGS = [
#     "Hey, this is a test of the speech synthesis system.",
#     "roupell received the announcement with a cheerful countenance.",
#     "with thirteen dollars, eighty-seven cents when considerably greater resources were available to him.",
#     "Two other witnesses were able to offer partial descriptions of a man they saw in the southeast corner window.",
#     "'just to steady their legs a little' in other words, to add his weight to that of the hanging bodies.",
#     "The discussion above has already set forth examples of his expression of hatred for the United States.",
#     "At two:thirty-eight p.m., Eastern Standard Time, Lyndon Baines Johnson took the oath of office as the thirty-sixth President of the United States.",
#     "or, quote, other high government officials in the nature of a complaint coupled with an expressed or implied determination to use a means.",
#     "As for my return entrance visa please consider it separately. End quote.",
#     "it appears that Marina Oswald also complained that her husband was not able to provide more material things for her.",
#     "appeared in The Dallas Times Herald on November fifteen, nineteen sixty-three.",
#     "The only exit from the office in the direction Oswald was moving was through the door to the front stairway.",
# ]

LIST_OF_TEST_STRINGS = ["您好！很高兴为您服务。"]


"""
buffer = []
for x in EncDecCTCModel.list_available_models():
    buffer.append(x.pretrained_model_name)
print(buffer)
buffer = []
for x in SpectrogramGenerator.list_available_models():
    buffer.append(x.pretrained_model_name)
print(buffer)
buffer = []
for x in Vocoder.list_available_models():
    buffer.append(x.pretrained_model_name)
print(buffer)

#################################################################################################################################################################################
ASR = ['QuartzNet15x5Base-En', 'stt_en_quartznet15x5', 'stt_en_jasper10x5dr', 'stt_ca_quartznet15x5', 
       'stt_it_quartznet15x5', 'stt_fr_quartznet15x5', 'stt_es_quartznet15x5', 'stt_de_quartznet15x5', 
       'stt_pl_quartznet15x5', 'stt_ru_quartznet15x5', 'stt_zh_citrinet_512', 'stt_zh_citrinet_1024_gamma_0_25', 
       'stt_zh_citrinet_1024_gamma_0_25', 'asr_talknet_aligner']

mel = ['tts_en_fastpitch', 'tts_en_fastpitch_ipa', 'tts_en_fastpitch_multispeaker', 'tts_de_fastpitch_singleSpeaker_thorstenNeutral_2102', 
       'tts_de_fastpitch_singleSpeaker_thorstenNeutral_2210', 'tts_de_fastpitch_multispeaker_5', 'tts_es_fastpitch_multispeaker', 
       'tts_zh_fastpitch_sfspeech', 'tts_en_fastpitch_for_asr_finetuning', 'tts_en_lj_mixertts', 'tts_en_lj_mixerttsx', 'tts_en_tacotron2']

TTS = ['tts_en_waveglow_88m', 'tts_en_hifigan', 'tts_en_lj_hifigan_ft_mixertts', 'tts_en_lj_hifigan_ft_mixerttsx', 
       'tts_en_hifitts_hifigan_ft_fastpitch', 'tts_de_hifigan_singleSpeaker_thorstenNeutral_2102', 'tts_de_hifigan_singleSpeaker_thorstenNeutral_2210', 
       'tts_de_hui_hifigan_ft_fastpitch_multispeaker_5', 'tts_es_hifigan_ft_fastpitch_multispeaker', 'tts_zh_hifigan_sfspeech', 'tts_en_lj_univnet', 'tts_en_libritts_univnet']
#################################################################################################################################################################################
"""

# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/checkpoints.html
# https://huggingface.co/models?library=pytorch,nemo&sort=trending
# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model",
        type=str,
        default="stt_zh_citrinet_512",
        choices=[x.pretrained_model_name for x in EncDecCTCModel.list_available_models()],
    )
    parser.add_argument(
        "--tts_model_spec",
        type=str,
        default="tts_zh_fastpitch_sfspeech",
        choices=[x.pretrained_model_name for x in SpectrogramGenerator.list_available_models()],
    )
    parser.add_argument(
        "--tts_model_vocoder",
        type=str,
        default="tts_zh_hifigan_sfspeech",
        choices=[x.pretrained_model_name for x in Vocoder.list_available_models()],
    )
    parser.add_argument("--wer_tolerance", type=float, default=1.0, help="used by test")
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    
    save_path = f'{args.tts_model_spec}+{args.tts_model_vocoder}+{args.asr_model}/'
    csv_file_path = save_path+f'{args.tts_model_spec}+{args.tts_model_vocoder}+{args.asr_model}.csv'
    if not os.path.exists(save_path):  
        os.makedirs(save_path)  

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)   
        
        if args.debug:
            logging.set_verbosity(logging.DEBUG)

        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        start = time.time()
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)
        end = time.time()
        asr_load_model = end -start
        logging.info(f"Using NGC cloud TTS Spectrogram Generator model {args.tts_model_spec}")
        start = time.time()
        tts_model_spec = SpectrogramGenerator.from_pretrained(model_name=args.tts_model_spec)
        end = time.time()
        tts_spec_load_model = end -start
        logging.info(f"Using NGC cloud TTS Vocoder model {args.tts_model_vocoder}")
        start = time.time()
        tts_model_vocoder = Vocoder.from_pretrained(model_name=args.tts_model_vocoder)
        end = time.time()
        tts_vocoder_load_model = end -start
        models = [asr_model, tts_model_spec, tts_model_vocoder]

        if torch.cuda.is_available():
            for i, m in enumerate(models):
                models[i] = m.cuda()
        for m in models:
            m.eval()

        asr_model, tts_model_spec, tts_model_vocoder = models

        parser = parsers.make_parser(
            labels=asr_model.decoder.vocabulary, name="en", unk_id=-1, blank_id=-1, do_normalize=True,
        )
        labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])

        tts_input = []
        asr_references = []
        longest_tts_input = 0
        for test_str in LIST_OF_TEST_STRINGS:
            tts_parsed_input = tts_model_spec.parse(test_str)
            if len(tts_parsed_input[0]) > longest_tts_input:
                longest_tts_input = len(tts_parsed_input[0])
            tts_input.append(tts_parsed_input.squeeze())

            asr_parsed = parser(test_str)
            asr_parsed = ''.join([labels_map[c] for c in asr_parsed])
            asr_references.append(asr_parsed)

        # Pad TTS Inputs
        for i, text in enumerate(tts_input):
            pad = (0, longest_tts_input - len(text))
            tts_input[i] = torch.nn.functional.pad(text, pad, value=68)

        logging.debug(tts_input)

        # Do TTS
        start = time.time()
        tts_input = torch.stack(tts_input)
        if torch.cuda.is_available():
            tts_input = tts_input.cuda()
        specs = tts_model_spec.generate_spectrogram(tokens=tts_input)
        end = time.time()
        tts_spec_time = end -start
        start = time.time()
        audio = []
        tts_time = []
        step = ceil(len(specs) / 12)
        for i in range(12):
            tts_start = time.time()
            audio.append(tts_model_vocoder.convert_spectrogram_to_audio(spec=specs[i * step : i * step + step]))
            tts_end = time.time()
            tts_time.append(tts_end - tts_start)
        end = time.time()
        tts_vocoder_time = end - start
        
        audio = [item for sublist in audio for item in sublist]
        audio_file_paths = []
        # Save audio
        logging.debug(f"args.trim: {args.trim}")
        for i, aud in enumerate(audio):
            aud = aud.cpu().numpy()
            if args.trim:
                aud = librosa.effects.trim(aud, top_db=40)[0]
            soundfile.write(save_path+f"{i}.wav", aud, samplerate=22050)
            audio_file_paths.append(save_path+f"{i}.wav")

        # Do ASR
        start = time.time()
        hypotheses = asr_model.transcribe(audio_file_paths)
        end = time.time()
        asr_time = end -start
        csv_writer.writerow(['ref', 'hyp', 'time'])
        for i, _ in enumerate(hypotheses):
            logging.debug(f"{i}")
            logging.debug(f"ref:'{asr_references[i]}'")
            logging.debug(f"hyp:'{hypotheses[i]}'")
            csv_writer.writerow([asr_references[i], hypotheses[i], tts_time[i]])
        wer_value = word_error_rate(hypotheses=hypotheses, references=asr_references)
        
        csv_writer.writerow(['WER', wer_value])
        csv_writer.writerow(['tts_spec_load_model', tts_spec_load_model])
        csv_writer.writerow(['tts_vocoder_load_model', tts_vocoder_load_model])
        csv_writer.writerow(['asr_load_model', asr_load_model])
        csv_writer.writerow(['tts_spec_time', tts_spec_time])
        csv_writer.writerow(['tts_vocoder_time', tts_vocoder_time])
        csv_writer.writerow(['asr_time', asr_time])
        
        if wer_value > args.wer_tolerance:
            raise ValueError(f"Got WER of {wer_value}. It was higher than {args.wer_tolerance}")
        logging.info(f'Got WER of {wer_value}. Tolerance was {args.wer_tolerance}')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
