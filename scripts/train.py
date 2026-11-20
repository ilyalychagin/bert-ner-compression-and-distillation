import torch
from seqeval.metrics import f1_score
from tqdm.auto import tqdm

from utils.utils import save_checkpoint


@torch.no_grad()
def validation(model, val_loader, criterion, device, metric=f1_score, epoch=None):
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    
    progress = tqdm(
        val_loader,
        desc=f"Validation epoch {epoch}" if epoch is not None else "Validation",
        unit="batch",
        dynamic_ncols=True,
        mininterval=2.0,
    )

    all_predicted_labels = []
    all_true_labels = []

    for batch in progress:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        output = model(**batch)
        predicted_ids = output.logits.argmax(dim=-1)

        for pred_row, true_row in zip(predicted_ids, batch["labels"]):
            keep = true_row != -100
            predicted_labels = [label_names[id_] for id_ in pred_row[keep]]
            true_labels = [label_names[id_] for id_  in true_row[keep]]
            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(true_labels)

        loss = criterion(
            output.logits.transpose(1, 2), 
            batch['labels']
        )

        num_tokens = (batch["labels"] != -100).sum().item()
        loss_value = loss.item()
        total_loss += loss_value * num_tokens 
        total_tokens += num_tokens

        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            avg_loss=f"{total_loss / total_tokens:.4f}",
            refresh=False,
        )

    
    return total_loss / total_tokens, metric(all_true_labels, all_predicted_labels)

def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device,
    checkpoint_dir=None,
    val_every_n_steps=25000,
):

    train_losses = []
    val_losses = []
    val_metrics = []
    global_step = 0
    start_epoch = 0
    best_val_f1 = float("-inf")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        progress = tqdm(
            train_loader,
            desc=f"Train epoch {epoch + 1}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=2.0,
        )
        for batch_index, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = criterion(output.logits.transpose(1, 2), batch["labels"])
            loss.backward()

            loss_value = loss.item()
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss_value * num_tokens
            total_tokens += num_tokens
            global_step += 1

            optimizer.step()
            optimizer.zero_grad()

            if global_step % val_every_n_steps == 0 or (
                batch_index == len(train_loader) and epoch + 1 == epochs
            ):
                val_loss, val_metric = validation(model, val_loader, criterion, device, epoch=epoch + 1)
                model.train()
                val_losses.append(val_loss)
                val_metrics.append(val_metric)

                print('F1-score:', val_metric)

                scheduler.step(val_loss)
                is_best = val_metric > best_val_f1
                if is_best:
                    best_val_f1 = val_metric

                    if checkpoint_dir is not None:
                        save_checkpoint(model, checkpoint_dir)


            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                avg_loss=f"{total_loss / total_tokens:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )

        epoch_loss = total_loss / total_tokens
        train_losses.append(epoch_loss)
        tqdm.write(
            f"Epoch {epoch + 1}/{epochs}: train_loss={epoch_loss:.4f}"
        )

    return train_losses, val_losses, val_metrics