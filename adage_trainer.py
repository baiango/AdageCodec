import time
import json


class AdageTrainer():
	def __init__(self):
		self.model = {}

	def train_mul_word(self, text):
		for i in range(len(text)):
			word = text[i]
			for j in range(i, len(text)):
				slice_ = text[i:j + 1]
				if len(slice_) == 1 or not any(c in {',', '.', '\n', ';', ':'} for c in slice_):
					self.model["".join(slice_)] = self.model.get("".join(slice_), 0) + 1
					for start in range(len(word)):
						subword = word[start:]
						self.model[subword] = self.model.get(subword, 0) + 1

	def align_text(self, text):
		'''Process text in reverse order and build token list'''
		tokens = []
		current_token = ''

		for char in reversed(list(text)):
			if char == ' ':
				tokens.append(char + current_token[::-1])
				current_token = ''
			elif char in {',', '.', '\n', '\'', '"', ';', ':'}:
				tokens.append(current_token[::-1])
				tokens.append(char)
				current_token = ''
			else:
				current_token += char

		return [t for t in tokens if t][::-1]

	def read_jsonl(self, file_path):
		with open(file_path, 'r') as f:
			for line in f:
				yield json.loads(line)['text']

	def train_model(self, text_path, clean=True, debug=False):
		'''This method trains the model using the given JSONL file at `text_path`. Training progress is printed every second, unless `debug` flag is set to True in which case training stops after processing the third example.'''
		jsonl_file = list(self.read_jsonl(text_path))
		total_time = 0
		prt_time = time.time()

		for (i, txt) in enumerate(jsonl_file):
			pre_text = self.align_text(txt)
			self.train_mul_word(pre_text)

			elapsed_time = time.time() - prt_time

			if int(elapsed_time) >= 1:
				total_time += time.time() - prt_time
				prt_time = time.time()
				print(f'Training: {i / len(jsonl_file) * 100:.2f}% {i / total_time:.1f} i/s')
			if debug and i == 3:
				break

		if clean:
			self.clean_model()

	def filter_min(self, threshold=2):
		'''Create a new dictionary containing only those keys whose corresponding value ≥ threshold'''
		self.model = {k: v for k, v in self.model.items() if v >= threshold}
		self.model["<unk>"] = 0

	def build_stats(self, token_byte_size=2):
		self.model = [(
			txt, [
				count * len(txt) - count * token_byte_size,
				count,
				index
			],
		) for index, (txt, count) in enumerate(self.model.items())]

	def sort_stats(self):
		fragments = {}
		for txt, stats in self.model:
			fragments.setdefault(len(txt), []).append([txt, stats])

		sort_high = lambda x: sorted(x, reverse=True, key=lambda x: x[0])
		self.model = sort_high([(txt, sort_high(stats)) for txt, stats in fragments.items()])

	def clean_model(self):
		self.filter_min()
		self.build_stats()
		self.sort_stats()

	def write_model(self, write_json_path):
		with open(write_json_path, 'w') as f:
			for length, vocab in self.model:
				json_string = json.dumps({
					'len': length,
					'vocab': [{
						'txt': txt,
						'gain': ga,
						'count': co,
						'index': i,
					} for txt, (ga, co, i) in vocab]})
				f.write(json_string + "\n")

if __name__ == '__main__':
	trainer = AdageTrainer()
	trainer.train_model("ai_poem.jsonl", debug=False)
	trainer.write_model('example_vocab.jsonl')
