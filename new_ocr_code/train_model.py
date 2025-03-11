import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Model_Train:
    def __init__(self, train_images_path, test_images_path, checkpoint_dir):
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path
        self.checkpoint_dir = checkpoint_dir
        self.img_width = 256
        self.img_height = 256
        self.num_classes = 38  # 0-9, A-Z, +, #
        self.hidden_size = 256
        self.embed_size = 256
        self.batch_size = 32
        self.lr = 3e-4
        self.n_epoch = 1000
        self.n_workers = 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_checkpoint_every = 30

        print("Initializing model and data loaders...")
        self.load_dataloader()
        self.load_model()

    def load_dataloader(self):
        self.ds_train = Container_Dataset(self.train_images_path, self.img_width, self.img_height, self.num_classes)
        self.ds_test = Container_Dataset(self.test_images_path, self.img_width, self.img_height, self.num_classes)
        self.tokenizer = self.ds_train.tokenizer
        self.train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
        self.test_loader = DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)

    def load_model(self):
        self.model = AttentionOCR(self.num_classes, self.hidden_size, self.embed_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_epoch(self):
        self.model.train()
        total_loss, total_acc, total_sentence_acc = 0, 0, 0
        for x, y in tqdm(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            loss, acc, sentence_acc = train_batch(x, y, self.model, self.optimizer, self.criterion, self.tokenizer)
            total_loss += loss
            total_acc += acc
            total_sentence_acc += sentence_acc
        return total_loss / len(self.train_loader), total_acc / len(self.train_loader), total_sentence_acc / len(self.train_loader)

    def eval_epoch(self):
        self.model.eval()
        total_loss, total_acc, total_sentence_acc = 0, 0, 0
        for x, y in tqdm(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)
            loss, acc, sentence_acc = eval_batch(x, y, self.model, self.criterion, self.tokenizer)
            total_loss += loss
            total_acc += acc
            total_sentence_acc += sentence_acc
        return total_loss / len(self.test_loader), total_acc / len(self.test_loader), total_sentence_acc / len(self.test_loader)

    def run(self):
        for epoch in range(self.n_epoch):
            train_loss, train_acc, train_sentence_acc = self.train_epoch()
            eval_loss, eval_acc, eval_sentence_acc = self.eval_epoch()
            if (epoch + 1) % 10 == 0:
              print(f"Epoch {epoch + 1}/{self.n_epoch}")
              print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Sentence Acc: {train_sentence_acc:.4f}")
              print(f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}, Eval Sentence Acc: {eval_sentence_acc:.4f}")

            if (epoch + 1) % self.save_checkpoint_every == 0:
                checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")


if __name__ == '__main__':
    train_images_path = "/content/drive/MyDrive/OCR_March/resized_images"
    test_images_path = "/content/drive/MyDrive/OCR_March/test"
    checkpoint_dir = "/content/drive/MyDrive/OCR_March/chkpoint"
    model_train = Model_Train(train_images_path, test_images_path, checkpoint_dir)
    model_train.run()
