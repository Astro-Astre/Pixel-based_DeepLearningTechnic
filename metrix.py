from args import *
import torch


def cf_metrics(loader, model, save: bool = False):
    y_pred = []  # save prediction
    y_true = []  # save ground truth
    device = "cuda:0"
    for inputs, labels in loader:
        inputs, label = inputs.to(data_config.device), inputs.to(data_config.device)
        output = model(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth
    classes = data_config.classes
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis],
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 10))
    if save:
        sns.heatmap(df_cm, annot=True).get_figure()
        plt.savefig(data_config.metrix_save_path)
    else:
        return sns.heatmap(df_cm, annot=True).get_figure()
