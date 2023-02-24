class model:
    def __call__(self, signal):
        inputs = self.processor(signal, 16000, return_tensors="pt")
        generated_ids = self.model.generate(
            inputs["input_features"], attention_mask=inputs["attention_mask"]
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def __init__(self):
        self.model = Speech2TextForConditionalGeneration.from_pretrained(
            "facebook/s2t-small-librispeech-asr"
        )
        self.processor = Speech2TextProcessor.from_pretrained(
            "facebook/s2t-small-librispeech-asr"
        )
