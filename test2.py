



import random

ENTAILMENT = 'entailment'
NON_ENTAILMENT = 'non_entailment'



class IntentExample:

    def __init__(self, text, label, do_lower_case):
        self.original_text = text
        self.text = text
        self.label = label

        if do_lower_case:
            self.text = self.text.lower()


class InputExample(object):

    def __init__(self, text_a, text_b, label = None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id




def load_intent_examples(file_path, do_lower_case):
    examples = []
    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            e = IntentExample(text.strip(), label.strip(), do_lower_case)
            examples.append(e)
    return examples


def load_intent_datasets(train_file_path, dev_file_path, do_lower_case):
    train_examples = load_intent_examples(train_file_path, do_lower_case)
    dev_examples = load_intent_examples(dev_file_path, do_lower_case)
    return train_examples, dev_examples



def sample(N, examples):
    labels = {} # unique classes

    for e in examples:
        if e.label in labels:
            labels[e.label].append(e.text)
        else:
            labels[e.label] = [e.text]

    sampled_examples = []
    for l in labels:
        random.shuffle(labels[l])
        if l == 'oos':
            examples = labels[l][:N]
        else:
            examples = labels[l][:N]
        sampled_examples.append({'task': l, 'examples': examples})

    return sampled_examples




def nli_base_example(tasks_data):
    # init
    all_entailment_examples = list()
    all_non_entailment_examples = list()
    # entailment
    for task in tasks_data:
        examples = task['examples']
        for j in range(len(examples)):
            for k in range(len(examples)):
                if k <= j:
                    continue
                all_entailment_examples.append(InputExample(examples[j], examples[k], ENTAILMENT))
                all_entailment_examples.append(InputExample(examples[k], examples[j], ENTAILMENT))
    # non entailment
    for task_1 in range(len(tasks_data)):
        for task_2 in range(len(tasks_data)):
            if task_2 <= task_1:
                continue
            examples_1 = tasks_data[task_1]['examples']
            examples_2 = tasks_data[task_2]['examples']
            for j in range(len(examples_1)):
                for k in range(len(examples_2)):
                    all_non_entailment_examples.append(InputExample(examples_1[j], examples_2[k], NON_ENTAILMENT))
                    all_non_entailment_examples.append(InputExample(examples_2[k], examples_1[j], NON_ENTAILMENT))  
    # combine
    nli_train_examples = all_entailment_examples + all_non_entailment_examples
    nli_dev_examples = all_entailment_examples[:100] + all_non_entailment_examples[:100]
    return nli_train_examples, nli_dev_examples

