import os
import torch
import pickle
import random
from eval import testing

def training(trainer, opt, train_dataset, test_dataset, batch_size, num_epoch, model_save=True, model_filename=None, logger=None, save_best=True, best_metric='NDCG10'):
    training_set_size = len(train_dataset)
    best_metric_value = 0.0
    best_epoch = -1
    
    for epoch in range(num_epoch):
        random.shuffle(train_dataset)
        num_batch = int(training_set_size / batch_size)
        a, b, c, d = zip(*train_dataset)
        trainer.train()
        all_C_distribs = []
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            train_loss, batch_C_distribs = trainer.global_update(supp_xs, supp_ys, query_xs, query_ys)
            all_C_distribs.append(batch_C_distribs)

        P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10 = testing(trainer, opt, test_dataset)
        logger.log(
            "{}\t{:.6f}\t TOP-5 {:.4f}\t{:.4f}\t{:.4f}\t TOP-7: {:.4f}\t{:.4f}\t{:.4f}"
            "\t TOP-10: {:.4f}\t{:.4f}\t{:.4f}".
                format(epoch, train_loss, P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10))
        
        # Get current metric value based on chosen metric
        current_metric_value = 0.0
        if best_metric == 'P5':
            current_metric_value = P5
        elif best_metric == 'NDCG5':
            current_metric_value = NDCG5
        elif best_metric == 'MAP5':
            current_metric_value = MAP5
        elif best_metric == 'P7':
            current_metric_value = P7
        elif best_metric == 'NDCG7':
            current_metric_value = NDCG7
        elif best_metric == 'MAP7':
            current_metric_value = MAP7
        elif best_metric == 'P10':
            current_metric_value = P10
        elif best_metric == 'NDCG10':
            current_metric_value = NDCG10
        elif best_metric == 'MAP10':
            current_metric_value = MAP10
        
        # Save best model if current performance is better
        if model_save and save_best and current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_epoch = epoch
            # Create best model filename by adding '_best' before the file extension
            base, ext = os.path.splitext(model_filename)
            best_model_filename = f"{base}_best{ext}"
            torch.save(trainer.state_dict(), best_model_filename)
            logger.log(f"New best model saved at epoch {epoch} with {best_metric} = {best_metric_value:.4f}")
            
        if epoch == (num_epoch-1):
            # Construct proper path using same directory as model_filename
            output_att_path = os.path.join(os.path.dirname(model_filename), 'output_att')
            with open(output_att_path, 'wb') as fp:
                pickle.dump(all_C_distribs, fp)

    # Save final model if requested (in addition to best model)
    if model_save and (not save_best or model_filename is not None):
        torch.save(trainer.state_dict(), model_filename)
        logger.log(f"Final model saved after {num_epoch} epochs")
    
    # Log information about best model
    if save_best and best_epoch >= 0:
        logger.log(f"Best model was from epoch {best_epoch} with {best_metric} = {best_metric_value:.4f}")
        
    return best_metric_value, best_epoch
