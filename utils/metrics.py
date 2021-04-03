import pandas as pd
import seaborn as sn
from matplotlib import style
import matplotlib.pyplot as plt
from time import sleep
from collections import defaultdict
from sklearn.metrics import auc

def confusion_matrix(y_actual, y_pred, title, subtitle = None, save_file=None):
	style.use('fivethirtyeight')

	data = {'y_actual': y_actual
			,'y_pred': y_pred
	}

	df = pd.DataFrame(data, columns=['y_actual','y_pred'])
	confusion_matrix = pd.crosstab(df['y_actual'], df['y_pred'], rownames=['Verdadeiro'], colnames=['Predição'])

	fig, ax = plt.subplots(figsize=(10,8))
	sn.heatmap(confusion_matrix, annot=True, ax=ax)

	ax.text(x=0.5, y=1.08, s=title, fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
	if subtitle is not None:
		ax.text(x=0.5, y=1.03, s=subtitle, fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)	
	
	if save_file is not None:
		fig.savefig(save_file)
		sleep(3)
		plt.close(fig)
	else:
		plt.show()


def get_accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


def get_evaluation_metrics(actual, predicted, pos_label=1):
	true_positives = 0
	true_negatives = 0
	false_positives = 0
	false_negatives = 0
	metrics = defaultdict()

	for i in range(len(actual)):
		if actual[i] == pos_label:
			if actual[i] == predicted[i]:
				true_positives += 1
			else:
				false_negatives += 1
		else:
			if actual[i] == predicted[i]:
				true_negatives += 1
			else:
				false_positives += 1

	metrics["accuracy"] = (true_positives + true_negatives) / float(len(actual))
	metrics["precision"] = true_positives / float(true_positives + false_positives)
	metrics["recall"] = true_positives / float(true_positives + false_negatives)
	metrics["f1_measure"] = (2 * metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
	metrics["true_positives"] = true_positives
	metrics["false_positives"] = false_positives
	metrics["true_negatives"] = true_negatives
	metrics["false_negatives"] = false_negatives
	return metrics
