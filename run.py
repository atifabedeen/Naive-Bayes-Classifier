from math import log10
import matplotlib.pyplot as plt
from utils import *
import pprint

def naive_bayes_train():
	percentage_positive_instances_train = 1
	percentage_negative_instances_train = 1

	percentage_positive_instances_test  = 1
	percentage_negative_instances_test  = 1

	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
	# with open('vocab.txt','w') as f:
	# 	for word in vocab:
	# 		f.write("%s\n" % word)
	# print("Vocabulary (training set):", len(vocab))

	print("Number of positive training instances:", len(pos_train))
	print("Number of negative training instances:", len(neg_train))
	print("Number of positive test instances:", len(pos_test))
	print("Number of negative test instances:", len(neg_test))
	pos_dict = {}
	pos_set = set()
	for review in pos_train:
		for i in range(len(review)):
			if review[i] not in pos_set:
				pos_set.add(review[i])
				pos_dict[review[i]] = 1
			else:
				pos_dict[review[i]] = pos_dict[review[i]] + 1 

	neg_dict = {}
	neg_set = set()
	for review in neg_train:
		for i in range(len(review)):
			if review[i] not in neg_set:
				neg_set.add(review[i])
				neg_dict[review[i]] = 1
			else:
				neg_dict[review[i]] = neg_dict[review[i]] + 1 
	
	return pos_dict, pos_set, neg_dict, neg_set, pos_train, neg_train, pos_test, neg_test, vocab
	

def naive_bayes(pos_dict, pos_set, neg_dict, neg_set, pos_train, neg_train, pos_test, neg_test):
	pr_pos_class = len(pos_train)/ (len(pos_train) + len(neg_train)) #Pr(y), y = 0
	pr_neg_class = len(neg_train)/ (len(pos_train) + len(neg_train)) #Pr(y), y = 1
	y_pos = [] #List of classes assigned by algorithm for each positive review in test set
	pos_sum = sum(len(row) for row in pos_train)
	neg_sum = sum(len(row) for row in neg_train)
	for review in pos_test:
		pr_pos = pr_pos_class
		pr_neg = pr_neg_class
		for word in set(review):
			if word in pos_set:
				pr_pos *= pos_dict[word]/pos_sum
			else:
				pr_pos = 0
				break
		for word in set(review):
			if word in neg_set:
				pr_neg *= neg_dict[word]/neg_sum
			else:
				pr_neg = 0
				break
		
		if (pr_pos == 0 and pr_neg == 0) or (pr_pos == pr_neg):
			y_pos.append(random.randint(0,1))

		elif pr_pos > pr_neg:
			y_pos.append(0) #positive class
		
		else:
			y_pos.append(1) #negative class)

	y_neg = [] #List of classes assigned by algorithm for each negative review in test set
	for review in neg_test:
		pr_pos = pr_pos_class
		pr_neg = pr_neg_class
		for word in set(review):
			if word in pos_set:
				pr_pos *= pos_dict[word]/pos_sum
			else:
				pr_pos = 0
				break
		for word in set(review):
			if word in neg_set:
				pr_neg *= neg_dict[word]/neg_sum
			else:
				pr_neg = 0
				break

		if (pr_pos == 0 and pr_neg == 0) or (pr_pos == pr_neg):
			y_neg.append(random.randint(0,1))

		elif pr_pos > pr_neg:
			y_neg.append(0) #positive class
		
		else:
			y_neg.append(1) #negative class
		

	confusion_matrix = [[y_pos.count(0), y_pos.count(1)], [y_neg.count(0), y_neg.count(1)]]
	print("confusion matrix", confusion_matrix)
	print("Accuracy", (y_pos.count(0) + y_neg.count(1))/(len(pos_test) + len(neg_test)))
	print("Precision", confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0]))
	print("Recall", confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1]))
	return


def naive_bayes_log(pos_dict, pos_set, neg_dict, neg_set, pos_train, neg_train, pos_test, neg_test, vocab):
	alpha = 10 #change to 0 if you want to use without laplace smoothing and comment a few lines below
	pr_pos_class = log10(len(pos_train)/ (len(pos_train) + len(neg_train))) #Pr(y), y = 0
	pr_neg_class = log10(len(neg_train)/ (len(pos_train) + len(neg_train))) #Pr(y), y = 1
	y_pos = [] #List of classes assigned by algorithm for each positive review in test set
	pos_sum = sum(len(row) for row in pos_train) + (alpha*len(vocab))
	neg_sum = sum(len(row) for row in neg_train) + (alpha*len(vocab))
	for review in pos_test:
		pr_pos = pr_pos_class
		pr_neg = pr_neg_class
		for word in set(review):
			if word in pos_set:
				pr_pos += log10((pos_dict[word]+alpha)/pos_sum)
			else:
				pr_pos += log10((alpha)/pos_sum) #Comment this else when you want to use without laplace smoothing

		for word in set(review):
			if word in neg_set:
				pr_neg += log10((neg_dict[word]+alpha)/neg_sum)
			else:
				pr_neg += log10((alpha)/neg_sum) #Comment this else when you want to use without laplace smoothing

		if (pr_pos == 0 and pr_neg == 0) or (pr_pos == pr_neg):
			y_pos.append(random.randint(0,1))

		elif pr_pos > pr_neg:
			y_pos.append(0) #positive class
		
		else:
			y_pos.append(1) #negative class


	y_neg = [] #List of classes assigned by algorithm for each negative review in test set
	for review in neg_test:
		pr_pos = pr_pos_class
		pr_neg = pr_neg_class
		for word in set(review):
			if word in pos_set:
				pr_pos += log10((pos_dict[word]+alpha)/pos_sum)
			else:
				pr_pos += log10((alpha)/pos_sum) #Comment this else when you want to use without laplace smoothing

		for word in set(review):
			if word in neg_set:
				pr_neg += log10((neg_dict[word]+alpha)/neg_sum)
			else:
				pr_neg += log10((alpha)/neg_sum) #Comment this else when you want to use without laplace smoothing

		if (pr_pos == 0 and pr_neg == 0) or (pr_pos == pr_neg):
			y_neg.append(random.randint(0,1))

		elif pr_pos > pr_neg:
			y_neg.append(0) #positive class
		
		else:
			y_neg.append(1) #negative class
		
	confusion_matrix = [[y_pos.count(0), y_pos.count(1)], [y_neg.count(0), y_neg.count(1)]]
	print("confusion matrix", confusion_matrix)
	print("Accuracy", (y_pos.count(0) + y_neg.count(1))/(len(pos_test) + len(neg_test)))
	print("Precision", confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0]))
	print("Recall", confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1]))		

	
def naive_bayes_alpha(pos_dict, pos_set, neg_dict, neg_set, pos_train, neg_train, pos_test, neg_test, vocab):
	pr_pos_class = log10(len(pos_train)/ (len(pos_train) + len(neg_train))) #Pr(y), y = 0
	pr_neg_class = log10(len(neg_train)/ (len(pos_train) + len(neg_train))) #Pr(y), y = 1
	alpha_lst = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
	acc = []
	for alpha in alpha_lst:
		y_pos = [] #List of classes assigned by algorithm for each positive review in test set
		pos_sum = sum(len(row) for row in pos_train) + (alpha*len(vocab))
		neg_sum = sum(len(row) for row in neg_train) + (alpha*len(vocab))
		for review in pos_test:
			pr_pos = pr_pos_class
			pr_neg = pr_neg_class
			for word in set(review):
				if word in pos_set:
					pr_pos += log10((pos_dict[word]+alpha)/pos_sum)
				else:
					pr_pos += log10((alpha)/pos_sum) 

			for word in set(review):
				if word in neg_set:
					pr_neg += log10((neg_dict[word]+alpha)/neg_sum)
				else:
					pr_neg += log10((alpha)/neg_sum) 

			if (pr_pos == 0 and pr_neg == 0) or (pr_pos == pr_neg):
				y_pos.append(random.randint(0,1))

			elif pr_pos > pr_neg:
				y_pos.append(0) #positive class
		
			else:
				y_pos.append(1) #negative class


		y_neg = [] #List of classes assigned by algorithm for each negative review in test set
		for review in neg_test:
			pr_pos = pr_pos_class
			pr_neg = pr_neg_class
			for word in set(review):
				if word in pos_set:
					pr_pos += log10((pos_dict[word]+alpha)/pos_sum)
				else:
					pr_pos += log10((alpha)/pos_sum) 

			for word in set(review):
				if word in neg_set:
					pr_neg += log10((neg_dict[word]+alpha)/neg_sum)
				else:
					pr_neg += log10((alpha)/neg_sum) 

			if (pr_pos == 0 and pr_neg == 0) or (pr_pos == pr_neg):
				y_neg.append(random.randint(0,1))

			elif pr_pos > pr_neg:
				y_neg.append(0) #positive class
		
			else:
				y_neg.append(1) #negative class
		
		print("When Alpha is: ", alpha, "Accuracy: ", (y_pos.count(0) + y_neg.count(1))/(len(pos_test) + len(neg_test)))
		acc.append((y_pos.count(0) + y_neg.count(1))/(len(pos_test) + len(neg_test)))

	plt.plot(alpha_lst, acc)
	plt.xlabel("Alpha values")
	plt.ylabel("Accuracy")
	plt.xscale("log")
	plt.show()
	return

if __name__=="__main__":
	print("Split into train and test datasets and train...")
	pos_dict, pos_set, neg_dict, neg_set, pos_train, neg_train, pos_test, neg_test, vocab = naive_bayes_train()
	print("Naive Bayes...")
	naive_bayes(pos_dict, pos_set, neg_dict, neg_set, pos_train, neg_train, pos_test, neg_test)
	print("Naive Bayes using log transformation...")
	naive_bayes_log(pos_dict, pos_set, neg_dict, neg_set, pos_train, neg_train, pos_test, neg_test, vocab)
	print("Naive Bayes with laplace smoothing...")
	naive_bayes_alpha(pos_dict, pos_set, neg_dict, neg_set, pos_train, neg_train, pos_test, neg_test, vocab)


