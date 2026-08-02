import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import os

def train(
    model,
    tensor_loader,
    val_loader,
    num_epochs,
    learning_rate,
    criterion,
    device,
    subset_ratio=0.4,
):
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    log_file = "training_log.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Training Log\n")
        f.write("=" * 100 + "\n\n")

    best_acc = 0.0

    for epoch in range(num_epochs):

        # ===========================
        # Train
        # ===========================
        model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_f1_scores = []

        for batch_idx, (inputs, labels) in enumerate(
            tqdm(tensor_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # ---------------------------
            # NaN check
            # ---------------------------
            if torch.isnan(loss):

                print("=" * 80)
                print("NaN detected")
                print("Epoch :", epoch + 1)
                print("Batch :", batch_idx)

                print("\nInput")
                print("shape :", inputs.shape)
                print("min   :", inputs.min().item())
                print("max   :", inputs.max().item())
                print("mean  :", inputs.mean().item())
                print("std   :", inputs.std().item())
                print("NaN   :", torch.isnan(inputs).any().item())

                print("\nOutput")
                print("shape :", outputs.shape)
                print("min   :", outputs.min().item())
                print("max   :", outputs.max().item())
                print("NaN   :", torch.isnan(outputs).any().item())

                for name, p in model.named_parameters():

                    if torch.isnan(p).any():
                        print(f"NaN weight : {name}")

                    if p.grad is not None:

                        if torch.isnan(p.grad).any():
                            print(f"NaN grad   : {name}")

                        if torch.isinf(p.grad).any():
                            print(f"Inf grad   : {name}")

                raise RuntimeError("Training stopped because loss became NaN.")

            loss.backward()

            # Uncomment nếu muốn dùng gradient clipping
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     max_norm=1.0
            # )

            optimizer.step()

            # kiểm tra weight sau khi update
            for name, p in model.named_parameters():

                if torch.isnan(p).any():
                    print(f"NaN weight after optimizer.step(): {name}")
                    raise RuntimeError("Weight became NaN.")

            epoch_loss += loss.item() * inputs.size(0)

            predict_y = torch.argmax(outputs, dim=1)

            epoch_accuracy += (
                (predict_y == labels).sum().item() / labels.size(0)
            )

            epoch_f1_scores.append(
                f1_score(
                    labels.cpu().numpy(),
                    predict_y.cpu().numpy(),
                    average="weighted",
                )
            )

        epoch_loss /= len(tensor_loader.dataset)
        epoch_accuracy /= len(tensor_loader)
        epoch_f1_score = sum(epoch_f1_scores) / len(epoch_f1_scores)

        # ===========================
        # Validation
        # ===========================
        model.eval()

        val_loss = 0.0
        val_accuracy = 0.0
        val_f1_scores = []

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.long().to(device, non_blocking=True)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                predict_y = torch.argmax(outputs, dim=1)

                val_accuracy += (
                    (predict_y == labels).sum().item() / labels.size(0)
                )

                val_f1_scores.append(
                    f1_score(
                        labels.cpu().numpy(),
                        predict_y.cpu().numpy(),
                        average="weighted",
                    )
                )

        val_loss /= len(val_loader.dataset)
        val_accuracy /= len(val_loader)
        val_f1_score = sum(val_f1_scores) / len(val_f1_scores)

        # ===========================
        # Save best model
        # ===========================
        if val_accuracy > best_acc:

            best_acc = val_accuracy

            torch.save(
                model.state_dict(),
                "best_model.pth",
            )

            print(
                f"Best model saved "
                f"(Epoch {epoch+1}, "
                f"Val Acc = {best_acc:.4f})"
            )

        log = (
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {epoch_loss:.9f} | "
            f"Train Acc: {epoch_accuracy:.4f} | "
            f"Train F1: {epoch_f1_score:.4f} | "
            f"Val Loss: {val_loss:.9f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1_score:.4f}"
        )

        print(log)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log + "\n")

        print(f"\nTraining log saved to: {os.path.abspath(log_file)}")

def test(model, tensor_loader, criterion, device):

    model.eval()

    test_acc = 0.0
    test_loss = 0.0
    test_f1_scores = []

    with torch.no_grad():

        for inputs, labels in tqdm(tensor_loader, desc="Testing"):

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            predict_y = torch.argmax(outputs, dim=1)

            accuracy = (
                (predict_y == labels).sum().item()
                / labels.size(0)
            )

            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)

            test_f1_scores.append(
                f1_score(
                    labels.cpu().numpy(),
                    predict_y.cpu().numpy(),
                    average="weighted",
                )
            )

    test_acc /= len(tensor_loader)
    test_loss /= len(tensor_loader.dataset)
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

def val(model, tensor_loader, criterion, device):

    model.eval()

    test_acc = 0.0
    test_loss = 0.0

    with torch.no_grad():

        for inputs, labels in tqdm(tensor_loader, desc="Validation"):

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            predict_y = torch.argmax(outputs, dim=1)

            accuracy = (
                (predict_y == labels).sum().item()
                / labels.size(0)
            )

            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)

    test_acc /= len(tensor_loader)
    test_loss /= len(tensor_loader.dataset)

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