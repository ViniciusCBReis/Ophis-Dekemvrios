import matplotlib.pyplot as plt

def plot_precision_recall(metrics):
    classes = list(metrics.keys())
    precisions = [metrics[cls]["precision"] for cls in classes]
    recalls = [metrics[cls]["recall"] for cls in classes]

    fig, ax = plt.subplots()
    ax.barh(classes, precisions, color='skyblue', label='Precision')
    ax.barh(classes, recalls, color='salmon', alpha=0.6, label='Recall')

    ax.set_xlabel('Scores')
    ax.set_title('Precision and Recall by Class')
    ax.legend()
    plt.show()

def calculate_f1(precision, recall):

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate(metrics): #I need looking with attention here
    for cls in metrics:
        precision = metrics[cls]["precision"]
        recall = metrics[cls]["recall"]
        metrics[cls]["f1"] = calculate_f1(precision, recall)

def plot_f1(metrics):
    classes = list(metrics.keys())
    f1_scores = [metrics[cls]["f1"] for cls in classes]

    plt.barh(classes, f1_scores, color='purple')
    plt.xlabel("F1 Score")
    plt.title("F1 Score by Class")
    plt.show()

def plot_loss(loss_per_epoch):
    epochs = list(range(1, len(loss_per_epoch) + 1))

    plt.plot(epochs, loss_per_epoch, marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.show()



