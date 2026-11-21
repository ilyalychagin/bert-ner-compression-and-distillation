import torch
from seqeval.metrics import f1_score
from tqdm.auto import tqdm
import torch.nn.functional as F

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
    teacher=None,
    T=1,
    alpha_soft=1,
    alpha_hard=1,
    scheduler_per_step=False,
    hidden_distillation=False,
    hidden_projection=None,
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
            output = model(**batch, output_hidden_states=hidden_distillation)
            if teacher:
                with torch.no_grad():
                    output_teacher = teacher(**batch, output_hidden_states=hidden_distillation)

            distillation_losses: dict[str, torch.Tensor] = {}

            loss_hard = criterion(output.logits.transpose(1, 2), batch["labels"])
            loss_teacher = loss_hard.new_zeros(())
            loss = alpha_hard * loss_hard

            if teacher:
                # teacher.eval() не забыть

                keep = batch["labels"] != -100

                with torch.no_grad():
                    teacher_logits = output_teacher.logits
                    soft_labels = torch.softmax(teacher_logits[keep]/T, dim=-1)

                student_logits = output.logits

                student_log_probs = torch.log_softmax(student_logits[keep] / T, dim=-1)

                loss_teacher = torch.nn.KLDivLoss(reduction="batchmean")(student_log_probs, soft_labels)
                loss = loss + alpha_soft * T**2 * loss_teacher
            
            if hidden_distillation:

                valid_tokens = batch["attention_mask"].bool()
                valid_pairs = valid_tokens[:, None, :, None] & valid_tokens[:, None, None, :]

                teacher_hidden_states = output_teacher.hidden_states
                student_hidden_states = output.hidden_states

                t_batch_size, t_num_len, _ = teacher_hidden_states[0].shape
                batch_size, num_len, _ = student_hidden_states[0].shape

                t_num_heads = teacher.config.num_attention_heads
                num_heads = model.config.num_attention_heads
                num_teacher_layers = len(teacher_hidden_states) - 1
                num_student_layers = len(student_hidden_states) - 1

                attn_loss = 0
                hidden_loss = 0
                for i in range(num_student_layers):
                    t = round((i + 1) * num_teacher_layers / num_student_layers) - 1

                    teacher_input = teacher_hidden_states[t]
                    student_input = student_hidden_states[i]
                    teacher_hidden_state = teacher_hidden_states[t + 1]
                    student_hidden_state = student_hidden_states[i + 1]

                    teacher_attention = teacher.bert.encoder.layer[t].attention.self
                    student_attention = model.bert.encoder.layer[i].attention.self

                    with torch.no_grad():
                        q_teacher = teacher_attention.query(teacher_input).reshape(
                            t_batch_size, t_num_len, t_num_heads, -1
                        ).transpose(1, 2)
                        k_teacher = teacher_attention.key(teacher_input).reshape(
                            t_batch_size, t_num_len, t_num_heads, -1
                        ).transpose(1, 2)
                        t_scores = (q_teacher @ k_teacher.transpose(-2, -1)) / q_teacher.shape[-1]**0.5

                    q_student = student_attention.query(student_input).reshape(
                        batch_size, num_len, num_heads, -1
                    ).transpose(1, 2)
                    k_student = student_attention.key(student_input).reshape(
                        batch_size, num_len, num_heads, -1
                    ).transpose(1, 2)
                    scores = (q_student @ k_student.transpose(-2, -1)) / q_student.shape[-1]**0.5

                    attn_loss += F.mse_loss(
                        scores.masked_select(valid_pairs),
                        t_scores.masked_select(valid_pairs),
                    )
                    hidden_loss += F.mse_loss(
                        hidden_projection(student_hidden_state[valid_tokens]),
                        teacher_hidden_state[valid_tokens],
                    )

                loss = loss + attn_loss + hidden_loss

            
            distillation_losses = {
                "loss_teacher": f"{loss_teacher:.4f}" if teacher is not None else "-",
                "hidden_loss": f"{hidden_loss:.4f}" if hidden_distillation else "-",
                "attn_loss": f"{attn_loss:.4f}" if hidden_distillation else "-",
                "loss_hard": f"{loss_hard:.4f}",
            }


            loss.backward()

            loss_value = loss.item()
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss_value * num_tokens
            total_tokens += num_tokens
            global_step += 1

            optimizer.step()
            optimizer.zero_grad()

            if scheduler_per_step:
                scheduler.step()

            if global_step % val_every_n_steps == 0 or (
                batch_index == len(train_loader) and epoch + 1 == epochs
            ):
                val_loss, val_metric = validation(model, val_loader, criterion, device, epoch=epoch + 1)
                model.train()
                val_losses.append(val_loss)
                val_metrics.append(val_metric)

                print('F1-score:', val_metric)

                if scheduler_per_step == False:
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
                **distillation_losses,
            )

        epoch_loss = total_loss / total_tokens
        train_losses.append(epoch_loss)
        tqdm.write(
            f"Epoch {epoch + 1}/{epochs}: train_loss={epoch_loss:.4f}"
        )

    return train_losses, val_losses, val_metrics