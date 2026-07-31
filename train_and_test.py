import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import os


def train(model, tensor_loader, val_loader, num_epochs, learning_rate,
          criterion, device, subset_ratio=0.4):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # =========================
    # Create log file
    # =========================
    log_file = "training_log.csv"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            "Epoch,"
            "TrainLoss,TrainAcc,TrainF1,"
            "ValLoss,ValAcc,ValF1\n"
        )

    num_samples = len(tensor_loader.dataset)
    indices = list(range(num_samples))

    for epoch in range(num_epochs):

        ############################
        # Training
        ############################
        model.train()

        epoch_loss = 0
        epoch_accuracy = 0
        epoch_f1_scores = []

        for data in tqdm(tensor_loader,
                         desc=f"Epoch {epoch + 1}/{num_epochs}"):

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

            predict_y = torch.argmax(outputs, dim=1).to(device)

            epoch_accuracy += (
                predict_y == labels.to(device)
            ).sum().item() / labels.size(0)

            f1 = f1_score(
                labels.cpu(),
                predict_y.cpu(),
                average="weighted"
            )

            epoch_f1_scores.append(f1)

        epoch_loss /= len(tensor_loader.dataset)
        epoch_accuracy /= len(tensor_loader)
        epoch_f1_score = sum(epoch_f1_scores) / len(epoch_f1_scores)

        ############################
        # Validation
        ############################
        model.eval()

        val_loss = 0
        val_accuracy = 0
        val_f1_scores = []

        with torch.no_grad():

            for data in val_loader:

                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.type(torch.LongTensor)

                outputs = model(inputs)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                predict_y = torch.argmax(outputs, dim=1)

                val_accuracy += (
                    predict_y == labels
                ).sum().item() / labels.size(0)

                f1 = f1_score(
                    labels.cpu(),
                    predict_y.cpu(),
                    average="weighted"
                )

                val_f1_scores.append(f1)

        val_loss /= len(val_loader.dataset)
        val_accuracy /= len(val_loader)
        val_f1_score = sum(val_f1_scores) / len(val_f1_scores)

        ############################
        # Print log
        ############################
        log = (
            f'Epoch [{epoch + 1}/{num_epochs}] | '
            f'Train Loss: {epoch_loss:.9f} | '
            f'Train Acc: {epoch_accuracy:.4f} | '
            f'Train F1: {epoch_f1_score:.4f} | '
            f'Val Loss: {val_loss:.9f} | '
            f'Val Acc: {val_accuracy:.4f} | '
            f'Val F1: {val_f1_score:.4f}'
        )

        print(log)

        ############################
        # Save log to file
        ############################
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1},"
                f"{epoch_loss:.9f},"
                f"{epoch_accuracy:.6f},"
                f"{epoch_f1_score:.6f},"
                f"{val_loss:.9f},"
                f"{val_accuracy:.6f},"
                f"{val_f1_score:.6f}\n"
            )

    print(f"\nTraining log saved to: {os.path.abspath(log_file)}")
    return