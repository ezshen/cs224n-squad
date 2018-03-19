import matplotlib.pyplot as plt

answer_train_file = '../data/train.answer'
answer_dev_file = '../data/dev.answer'

context_train_file = '../data/train.context'
context_dev_file = '../data/dev.context'

question_train_file = '../data/train.question'
question_dev_file = '../data/dev.question'

def plot_lens(a, filename, title, x, y):
	plt.figure()
	plt.title(title)
	plt.ylabel(y)
	plt.xlabel(x)
	plt.hist(a, 25)
	plt.savefig('../figures/' + filename)

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

who_c, who_q, who_a = [], [], []
what_c, what_q, what_a = [], [], []
when_c, when_q, when_a = [], [], []
where_c, where_q, where_a = [], [], []
why_c, why_q, why_a = [], [], []
how_c, how_q, how_a = [], [], []
which_c, which_q, which_a = [], [], []
other_c, other_q, other_a = [], [], []

for i in range(len(question)):
	q = question[i]
	isQuestion = False 
	for j in range(len(q)):
		if q[j] == 'who':
			who_c.append(context[i])
			who_q.append(q)
			who_a.append(answer[i])
			isQuestion = True
			break
		elif q[j] == 'what':
			what_c.append(context[i])
			what_q.append(q)
			what_a.append(answer[i])
			isQuestion = True
			break
		elif q[j] == 'when':
			when_c.append(context[i])
			when_q.append(q)
			when_a.append(answer[i])
			isQuestion = True
			break
		elif q[j] == 'where':
			where_c.append(context[i])
			where_q.append(q)
			where_a.append(answer[i])
			isQuestion = True
			break
		elif q[j] == 'why':
			why_c.append(context[i])
			why_q.append(q)
			why_a.append(answer[i])
			isQuestion = True
			break
		elif q[j] == 'how':
			how_c.append(context[i])
			how_q.append(q)
			how_a.append(answer[i])
			isQuestion = True
			break
		elif q[j] == 'which':
			which_c.append(context[i])
			which_q.append(q)
			which_a.append(answer[i])
			isQuestion = True
			break
	if not isQuestion:
		other_c.append(context[i])
		other_q.append(q)
		other_a.append(answer[i])

who_answer_lens = [len(x) for x in who_a]
what_answer_lens = [len(x) for x in what_a]
when_answer_lens = [len(x) for x in when_a]
where_answer_lens = [len(x) for x in where_a]
why_answer_lens = [len(x) for x in why_a]
how_answer_lens = [len(x) for x in how_a]
which_answer_lens = [len(x) for x in which_a]

plot_lens(answer_lens, 'answer_lens.png', 'Answer Lengths for Train and Dev', 'Length', 'Count')
plot_lens(context_lens, 'context_lens.png', 'Context Lengths for Train and Dev', 'Length', 'Count') 
plot_lens(question_lens, 'question_lens.png', 'Question Lengths for Train and Dev', 'Length', 'Count')
plot_lens(who_answer_lens, 'who_answer_lens.png', 'Answer Lengths for Who Questions', 'Length', 'Count')
plot_lens(what_answer_lens, 'what_answer_lens.png', 'Answer Lengths for What Questions', 'Length', 'Count')
plot_lens(when_answer_lens, 'when_answer_lens.png', 'Answer Lengths for When Questions', 'Length', 'Count')
plot_lens(where_answer_lens, 'where_answer_lens.png', 'Answer Lengths for Where Questions', 'Length', 'Count')
plot_lens(why_answer_lens, 'why_answer_lens.png', 'Answer Lengths for Why Questions', 'Length', 'Count')
plot_lens(how_answer_lens, 'how_answer_lens.png', 'Answer Lengths for How Questions', 'Length', 'Count')
plot_lens(which_answer_lens, 'which_answer_lens.png', 'Answer Lengths for Which Questions', 'Length', 'Count')



