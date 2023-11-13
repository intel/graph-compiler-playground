import torch


def train(
    model, dtype, device, train_loader, criterion, optimizer, epoch, log_interval=200
):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)
        data = data.type(dtype)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)[0]

        # Calculate loss
        output = output.type(torch.float)
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
