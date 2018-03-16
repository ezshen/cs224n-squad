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

tot_examples = len(answer)

answer_lens = [len(x) for x in answer]
context_lens = [len(x) for x in context]
question_lens = [len(x) for x in question]

who_c, who_q, who_a = [], [], []
what_c, what_q, what_a = [], [], []
when_c, when_q, when_a = [], [], []
where_c, where_q, where_a = [], [], []
why_c, why_q, why_a = [], [], []
how_c, how_q, how_a = [], [], []
other_c, other_q, other_a = [], [], []


for i in range(tot_examples):
	categorized = False
	curr_q = question[i]
	for j in range(len(curr_q)):
		if curr_q[j] == 'who':
			who_c.append(context[i])
			who_q.append(curr_q)
			who_a.append(answer[i])
			categorized = True
			break
		elif curr_q[j] == 'what':
			what_c.append(context[i])
			what_q.append(curr_q)
			what_a.append(answer[i])
			categorized = True
			break
		elif curr_q[j] == 'when':
			when_c.append(context[i])
			when_q.append(curr_q)
			when_a.append(answer[i])
			categorized = True
			break
		elif curr_q[j] == 'where':
			where_c.append(context[i])
			where_q.append(curr_q)
			where_a.append(answer[i])
			categorized = True
			break
		elif curr_q[j] == 'why':
			why_c.append(context[i])
			why_q.append(curr_q)
			why_a.append(answer[i])
			categorized = True
			break
		elif curr_q[j] == 'how':
			how_c.append(context[i])
			how_q.append(curr_q)
			how_a.append(answer[i])
			categorized = True
			break
	if not categorized:
		other_c.append(context[i])
		other_q.append(curr_q)
		other_a.append(answer[i])


who_answer_len = [len(x) for x in who_a]
what_answer_len = [len(x) for x in what_a]
when_answer_len = [len(x) for x in when_a]
where_answer_len = [len(x) for x in where_a]
why_answer_len = [len(x) for x in why_a]
how_answer_len = [len(x) for x in how_a]
other_answer_len = [len(x) for x in other_a]

print "percentage of who questions:" + str(len(who_q)/float(tot_examples))
print "percentage of what questions:" + str(len(what_q)/float(tot_examples))
print "percentage of when questions:" + str(len(when_q)/float(tot_examples))
print "percentage of where questions:" + str(len(where_q)/float(tot_examples))
print "percentage of why questions:" + str(len(why_q)/float(tot_examples))
print "percentage of how questions:" + str(len(how_q)/float(tot_examples))
print "percentage of other questions:" + str(len(other_q)/float(tot_examples))

def plot_lens(a, filename):
    plt.figure()
    plt.hist(a, 25)
    plt.savefig('../figures/' + filename)


plot_lens(who_answer_len, 'who_answer_lens.png')
plot_lens(what_answer_len, 'what_answer_lens.png')
plot_lens(when_answer_len, 'when_answer_lens.png')
plot_lens(where_answer_len, 'where_answer_lens.png')
plot_lens(why_answer_len, 'why_answer_lens.png')
plot_lens(how_answer_len, 'how_answer_lens.png')
plot_lens(other_answer_len, 'other_answer_lens.png')
#plot_lens(answer_lens, 'answer_lens.png')
#plot_lens(context_lens, 'context_lens.png')
#plot_lens(question_lens, 'question_lens.png')



