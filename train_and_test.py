import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import os

# 定义训练函数
def train(model, tensor_loader, val_loader, num_epochs, learning_rate, criterion, device, subset_ratio=0.4):
    # 假设 CUDA 可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Log file
    log_file = "training_log.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Training Log\n")
        f.write("=" * 100 + "\n\n")

    num_samples = len(tensor_loader.dataset)
    indices = list(range(num_samples))

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_f1_scores = []

        for data in tqdm(tensor_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
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
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)

            # Calculate F1 score for the current batch
            f1 = f1_score(labels.cpu(), predict_y.cpu(), average='weighted')
            epoch_f1_scores.append(f1)

        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        epoch_f1_score = sum(epoch_f1_scores) / len(epoch_f1_scores)

        # 验证阶段
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

                print(torch.isnan(outputs).any())
                print(outputs.min().item())
                print(outputs.max().item())
                exit()

                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                predict_y = torch.argmax(outputs, dim=1)
                val_accuracy += (predict_y == labels).sum().item() / labels.size(0)

                # Calculate F1 score for the current batch
                f1 = f1_score(labels.cpu(), predict_y.cpu(), average='weighted')
                val_f1_scores.append(f1)

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_accuracy / len(val_loader)
        val_f1_score = sum(val_f1_scores) / len(val_f1_scores)

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

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log + "\n")

        print(f"\nTraining log saved to: {os.path.abspath(log_file)}")
    return

# 定义测试函数，输出acc，F1 score和loss
def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    test_f1_scores = []

    with torch.no_grad():
        for data in tqdm(tensor_loader, desc="Testing"):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)

            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)

            # Calculate F1 score for the current batch
            f1 = f1_score(labels.cpu(), predict_y.cpu(), average='weighted')
            test_f1_scores.append(f1)

    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    test_f1_score = sum(test_f1_scores) / len(test_f1_scores)

    log_file = "training_log.txt"

    log = (
        f"Final Test | "
        f"Loss: {test_loss:.9f} | "
        f"Acc: {test_acc:.4f} | "
        f"F1: {test_f1_score:.4f}"
    )

    print(log)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("FINAL TEST\n")
        f.write(log + "\n")
    return

# 备选测试函数，输出acc和loss
def val(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    with torch.no_grad():
        for data in tqdm(tensor_loader, desc="Testing"):
            inputs, labels = data
            inputs = inputs.to(device)
            labels.to(device)
            labels = labels.type(torch.LongTensor)

            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)

            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)
    
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    log_file = "training_log.txt"

    log = (
        f"Validation | "
        f"Loss: {test_loss:.9f} | "
        f"Acc: {test_acc:.4f}"
    )

    print(log)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log + "\n")
    return
