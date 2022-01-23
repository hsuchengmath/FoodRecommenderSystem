


import torch
from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
from datasets import load_dataset
import soundfile as sf

model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    print(batch["file"])
    batch["speech"] = speech
    return batch

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)
inputs = processor(ds["speech"][1], sampling_rate=16_000, return_tensors="pt")
#generated_ids = model.generate(input_ids=inputs["input_values"], attention_mask=inputs["attention_mask"])
generated_ids = model.generate(**inputs)

stranscription = processor.batch_decode(generated_ids)
print(stranscription)

