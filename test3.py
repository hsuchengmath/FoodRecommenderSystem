

'''
1. load data [DONE]
2. daya process to train_examples, dev_examples
3. code pipeline
'''


'''
train_examples (list)
self.text = text
self.label = label
----------------------
sampled_tasks : T->element(15-domain)->dict('task', 'examples')
task=> transfer
examples=> ['send $100 from paypal to my bank', 
            'transfer $10 from checking to savings', 
            'transfer between two accounts', 
            'send 400 dollars between city bank and usaa accounts', 
            'transfer $5 from savings to checking'] (5-shot)
'''

# daya process to train_examples, dev_examples
from util import load_intent_datasets, sample
path = 'clinc150/all/'
train_file_path = path + 'train'
test_file_path = path + 'test'
do_lower_case = True
train_examples, test_example = load_intent_datasets(train_file_path, test_file_path, do_lower_case)


# setting 5-shot data
from util import sample
N = 5
sampled_task = sample(N, train_examples)



from util import nli_base_example
nli_train_examples, nli_dev_examples = nli_base_example(sampled_task)



internal_config  = {
                    'bert_model' : 'roberta-base',
                    'pretrain_model_path' : None,
                    'intent_model_path' : 'bucket_model/downstream/',
                    'batch_size' : 400,
                    'gradient_accumulation_steps' : 4,
                    'seed' : 42,
                    'epoch' : 10,
                    'label_smoothing' : 0.1,
                    'max_grad_norm' : 1.0,
                    'learning_rate' : 2e-5,
                    'adam_epsilon' : 1e-8,
                    'warmup_proportion' : 0.1,
                    'max_seq_length' : 128
                    }


# train
from model import DNNC
model = DNNC(internal_config=internal_config)
model.train(nli_train_examples, nli_dev_examples)










