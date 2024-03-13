from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


def prepare_dataset(df):
    X = df.drop(columns=["failure"]).values
    y = df["failure"].values
    return X, y


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


def evaluate_model(y_true, y_pred, y_prob):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f'cal:{tn} {fp} {fn} {tp}')
    specificity = tn / (tn + fp)
    if y_prob.shape[1] == 1:  # 如果y_prob只有一列
        roc_auc = roc_auc_score(y_true, y_prob[:, 0])
    else:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    return precision, recall, specificity, f1, roc_auc
