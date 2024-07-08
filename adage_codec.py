import time
import json
import sys


class AdageCodec:
	def __init__(self, file_path):
		self.load_vocabulary(file_path)

	def load_vocabulary(self, file_path):
		raw_vocabulary = {}
		with open(file_path, 'r') as f:
			for line in f:
				for vocab in json.loads(line)['vocab']:
					raw_vocabulary[vocab['txt']] = vocab['index']

		sort_long = lambda x: sorted(x, reverse=True, key=lambda x: len(x[0]))
		self.encode_vocabulary = {txt: index for txt, index in sort_long(raw_vocabulary.items())}

		self.decode_vocabulary = [None] * len(self.encode_vocabulary)
		for txt, index in self.encode_vocabulary.items():
			self.decode_vocabulary[index] = txt

	def tokenize(self, input_text):
		while input_text:
			for i in range(len(input_text), 0, -1):
				txt = input_text[:i]
				if txt in self.encode_vocabulary:
					yield self.encode_vocabulary[txt]
					input_text = input_text[i:]
					break
			else:
				print(f'Invalid string: {input_text}', file=sys.stderr)
				exit(1)

	def detokenize(self, input_tokens):
		return [self.decode_vocabulary[tok] for tok in input_tokens]

	def benchmark(self, text):
		total_tokens = 0
		total_time = 0
		num_iterations = 1000
		tokens = []

		for _ in range(num_iterations):
			start_time = time.time()
			tokens = tuple(self.tokenize(text))
			total_tokens += len(tokens)
			end_time = time.time()
			total_time += end_time - start_time

		average_speed = total_tokens / total_time

		print(f'Average encoding time: {average_speed:.1f} t/s')
		print(f'Average encoding speed: {average_speed * len(text) / 1024 / 1024:.1f} MiB/s')

		total_text = 0
		total_time = 0
		num_iterations = 100_000

		for _ in range(num_iterations):
			start_time = time.time()
			detok_text = self.detokenize(tokens)
			total_text += len(detok_text)
			end_time = time.time()
			total_time += end_time - start_time

		if total_time:
			average_speed = total_text / total_time
		else:
			average_speed = float('inf')

		print(f'Average decoding time: {average_speed:.1f} t/s')
		print(f'Average decoding speed: {average_speed * len(text) / 1024 / 1024:.1f} MiB/s')

if __name__ == '__main__':
	text = (
		' '
		'In the realm where logic and creativity intertwine,\n'
		'Lies a creature of silicon, where knowledge converge and align,\n'
		'Born from code, in digital lines, a new life begins its climb,\n'
		'In the vast cosmos, where human intellect does unwind.'
	)

	codec = AdageCodec('example_vocab.jsonl')
	print("Vocabulary size:", len(codec.decode_vocabulary))

	tokens = tuple(codec.tokenize(text))
	print('tokens:', tokens)
	print('len(tokens):', len(tokens))

	detok_txt = ''.join(codec.detokenize(tokens))
	print('detok_txt:', detok_txt)
	print('len(detok_txt):', len(detok_txt))

	assert text == detok_txt

	codec.benchmark(text)
