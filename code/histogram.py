import matplotlib.pyplot as plt

answer_train_file = '../data/train.answer'
answer_dev_file = '../data/dev.answer'

context_train_file = '../data/train.context'
context_dev_file = '../data/dev.context'

question_train_file = '../data/train.question'
question_dev_file = '../data/dev.question'

with open(answer_train_file) as f:
    answer = f.readlines()

with open(answer_dev_file) as f:
    answer.extend(f.readlines())

with open(context_train_file) as f:
    context = f.readlines()

with open(context_dev_file) as f:
    context.extend(f.readlines())

with open(question_train_file) as f:
    question = f.readlines()

with open(question_dev_file) as f:
    question.extend(f.readlines())

answer = [x.strip().split(' ') for x in answer]
context = [x.strip().split(' ') for x in context]
question = [x.strip().split(' ') for x in question]

answer_lens = [len(x) for x in answer]
context_lens = [len(x) for x in context]
question_lens = [len(x) for x in question]

def plot_lens(a, filename):
    plt.figure()
    plt.hist(a, 25)
    plt.savefig('../figures/' + filename)

plot_lens(answer_lens, 'answer_lens.png')
plot_lens(context_lens, 'context_lens.png')
plot_lens(question_lens, 'question_lens.png')



