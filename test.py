import os
import torch
import torch.backends.cudnn as cudnn
from models.model import CNNModel  # Assuming this is where your model is defined
from sklearn.metrics import f1_score, recall_score

def test(dataset_name, epoch, test_loader):
    #assert dataset_name in ['data_private', 'mendeley_lbc']

    model_root = os.path.join('.', 'cache')

    cuda = True
    cudnn.benchmark = True
    alpha = 0

    """ testing """

    # Load the model
    my_net = CNNModel(num_classes=3) # Create a new instance of your model
    my_net.load_state_dict(torch.load(os.path.join(model_root, f'model_epoch_{epoch}.pth')))
    my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    n_total = 0
    n_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to track gradients during testing
        for t_img, t_label in test_loader:
            batch_size = len(t_label)

            if cuda:
                t_img = t_img.cuda()
                t_label = t_label.cuda()

            class_output, _ = my_net(input_data=t_img, alpha=alpha)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum().item()
            n_total += batch_size

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(t_label.cpu().numpy())

    accuracy = n_correct / n_total
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Change average method if needed
    recall = recall_score(all_labels, all_preds, average='weighted')  # Change average method if needed
    return accuracy, f1  # Return accuracy and F1-score for potential logging or early stopping
