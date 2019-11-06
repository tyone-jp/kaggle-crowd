import glob
import os
from sklearn.metrics import precision_recall_curve, auc
from keras.callbacks import Callback
import keras


class PrAucCallback(Callback):
    def __init__(self,
                 num_workers,
                 datagenerator,
                 early_stopping_patience=5,
                 plateau_patience=3,
                 reduction_rate=0.5,
                 stage='train',
                 checkpoints_path='./checkpoints/'):
        super(Callback, self).__init__()
        self.datagenerator = datagenerator
        self.num_workers = num_workers
        self.class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
        self.history = [[] for _ in range(len(self.class_names)+1)]
        self.early_stopping_patience = early_stopping_patience
        self.plateau_patience = plateau_patience
        self.reduction_rate = reduction_rate
        self.stage = stage
        self.best_pr_auc = -float('inf')
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        self.checkpoints_path = checkpoints_path

    def compute_pr_auc(self, y_true, y_pred):
        pr_auc_mean = 0
        print(f"\n{'#'*30}\n")
        for class_i in range(len(self.class_names)):
            precision, recall, _ = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
            pr_auc = auc(recall, precision)
            pr_auc_mean += pr_auc / len(self.class_names)
            print(f"PR AUC {self.class_names[class_i]},{self.stage}:{pr_auc:.3f}\n")
            self.history[class_i].append(pr_auc)
        print(f"\n{'#'*20}\n PR AUC mean,{self.stage}:{pr_auc_mean:.3f}\n{'#'*20}\n")
        self.history[-1].append(pr_auc_mean)
        return pr_auc_mean

    def is_patience_lost(self, patience):
        if len(self.history[-1]) > patience:
            best_performance = max(self.history[-1][-(patience+1):-1])
            return best_performance == self.history[-1][-(patience+1)]

    def early_stopping_check(self):
        if self.is_patience_lost(self.early_stopping_patience):
            self.model.stop_training = True

    def model_checkpoint(self, pr_auc_mean, epoch):
        if pr_auc_mean > self.best_pr_auc:
            for checkpoint in glob.glob(os.path.join(self.checkpoints_path, 'classifier_epoch_*')):
                os.remove(checkpoint)
            self.best_pr_auc = pr_auc_mean
            self.model.save(os.path.join(self.checkpoints_path, f'classifier_epoch_{epoch}_val_pr_auc_{pr_auc_mean:.3f}.h5'))
            print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")

    def reduce_lr_on_plateau(self):
        if self.is_patience_lost(self.plateau_patience):
            new_lr = float(keras.backend.get_value(self.model.optimizer.lr))*self.reduction_rate
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\n{'#'*20}\nReduced learning rate to {new_lr}.\n{'#'*20}\n")

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.datagenerator, workers=self.num_workers)
        y_true = self.datagenerator.get_labels()
        pr_auc_mean = self.compute_pr_auc(y_true, y_pred)

        if self.stage == 'val':
            self.early_stopping_check()
            self.model_checkpoint(pr_auc_mean, epoch)
            self.reduce_lr_on_plateau()

    def get_pr_auc_history(self):
        return self.history
