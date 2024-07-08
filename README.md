# 🚀 AgageCodec
## 🧨 Main features
- Handle multi-word, word, and subword text.
- Fast encoding speed 14.5 MiB/s (64902.5 t/s).
- Instant decoding speed 4995.6 MiB/s (22385810.8 t/s).

## ⭐ Drawbacks
- Training speed is 1462x slower or 0.0006% (35 KiB/s) compared to 50 MiB/s like [tokenizers](https://github.com/huggingface/tokenizers), slow training time.
- Unable to handle out-of-vocabulary text, so it throws ValueError.
- Runs in single-threaded only.
- ✨ Fully written in Python w/o dependency for extra slowness.

## ⚡️ Performance
| CPU      | Training | Encoding                 | Decoding                      |
|----------|----------|--------------------------|-------------------------------|
| i5-9300H | 35 KiB/s | 14.5 MiB/s (64902.5 t/s) | 4995.6 MiB/s (22385810.8 t/s) |

# 🔧 Usage
Please see [main.py](main.py) for an up-to-date full example code.

## 📊 To train
```py
import adage_trainer

trainer = adage_trainer.AdageTrainer()
trainer.train_model("ai_poem.jsonl")
trainer.write_model('example_vocab.jsonl')
```

## 📝 To load the vocabulary
```py
import adage_codec

codec = adage_codec.AdageCodec('example_vocab.jsonl')
print("Vocabulary size:", len(codec.decode_vocabulary))

text = (
	' '
	'In the realm where logic and creativity intertwine,\n'
	'Lies a creature of silicon, where knowledge converge and align,\n'
	'Born from code, in digital lines, a new life begins its climb,\n'
	'In the vast cosmos, where human intellect does unwind.'
)
```

## 🔑 To encode
```py
tokens = tuple(codec.tokenize(text))
print('tokens:', tokens)
print('len(tokens):', len(tokens))
```
## 🔓 To decode
```py
detok_txt = ''.join(codec.detokenize(tokens))
print('detok_txt:', detok_txt)
print('len(detok_txt):', len(detok_txt))
```

## 🏎️ To benchmark
```py
assert text == detok_txt
codec.benchmark(text)
```

# 🖊️ References
- Yang, J. (2024). Rethinking Tokenization: Crafting Better Tokenizers for Large Language Models. arXiv preprint arXiv:2403.00417. <https://arxiv.org/abs/2403.00417>
