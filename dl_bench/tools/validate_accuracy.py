import torch


def load_mnist_dataset():
    from torchvision import datasets, transforms

    batch_size = 32

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )

    validation_dataset = datasets.MNIST(
        "./data", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False
    )

    return (train_loader, validation_loader)


def validate_accuracy(
    model,
    dtype,
    criterion,
    validation_loader,
    device,
    loss_vector=[],
    accuracy_vector=[],
):
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device)
            target = target.to(device)
            # print("data: ", data, " device: ", device)
            data = data.type(dtype)

            output = model(data)[0]
            # print("target: ", target)
            # print("output: ", output)
            output = output.type(torch.float)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            val_loss, correct, len(validation_loader.dataset), accuracy
        )
    )


def validate_accuracy_mnist(
    model, dtype, criterion, device, loss_vector=[], accuracy_vector=[]
):
    _, validation_loader = load_mnist_dataset()
    validate_accuracy(
        model, dtype, criterion, validation_loader, device, loss_vector, accuracy_vector
    )
