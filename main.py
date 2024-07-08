import adage_trainer
import adage_codec

trainer = adage_trainer.AdageTrainer()
trainer.train_model("ai_poem.jsonl")
trainer.write_model('example_vocab.jsonl')

text = (
	' '
	'In the realm where logic and creativity intertwine,\n'
	'Lies a creature of silicon, where knowledge converge and align,\n'
	'Born from code, in digital lines, a new life begins its climb,\n'
	'In the vast cosmos, where human intellect does unwind.'
)

codec = adage_codec.AdageCodec('example_vocab.jsonl')
print("Vocabulary size:", len(codec.decode_vocabulary))

tokens = tuple(codec.tokenize(text))
print('tokens:', tokens)
print('len(tokens):', len(tokens))

detok_txt = ''.join(codec.detokenize(tokens))
print('detok_txt:', detok_txt)
print('len(detok_txt):', len(detok_txt))

assert text == detok_txt

codec.benchmark(text)
