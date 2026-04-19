from AI_Food_Recognition_Nutrition_Assistant import logger
from AI_Food_Recognition_Nutrition_Assistant.config.configuration import DataPreprocessingConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

class DataPreprocessing:
    """
        Define Transformers and augment them -> train test split
    """
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
    
    # Define transformers
    def get_train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomResizedCrop(self.config.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=self.config.randaugment_num_ops,
                magnitude=self.config.randaugment_magnitude
            ),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=self.config.random_erasing_p,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            ),
        ])
 
    def get_val_test_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.config.resize_size),
            transforms.CenterCrop(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
 
    def load_and_split(self):
        train_transform = self.get_train_transform()
        val_test_transform = self.get_val_test_transform()

        base_dataset = datasets.ImageFolder(root=self.config.unzip_dir)

        logger.info(f"Classes found: {len(base_dataset.classes)}")
        logger.info(f"Total images: {len(base_dataset)}")

        # define size
        train_size = int(0.8 * len(base_dataset))
        val_size = int(0.1 * len(base_dataset))
        test_size = len(base_dataset) - train_size - val_size
        
        # get seed and train-test split
        seed_num = torch.Generator().manual_seed(self.config.seed)
        train_split, val_split, test_split = random_split(
            base_dataset,
            lengths=[train_size,val_size,test_size],
            generator=seed_num
        )
        torch.save(
            {
            "train_indices": train_split.indices,
            "val_indices": val_split.indices,
            "test_indices": test_split.indices,
            },
            self.config.splits_dir
        )
        # create separate datasets so each split can have its own transform
        train_dataset = datasets.ImageFolder(
            root=self.config.unzip_dir,
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=self.config.unzip_dir,
            transform=val_test_transform
        )
        test_dataset = datasets.ImageFolder(
            root=self.config.unzip_dir,
            transform=val_test_transform
        )

        # reuse indices from the split
        train_data = torch.utils.data.Subset(train_dataset, train_split.indices)
        val_data = torch.utils.data.Subset(val_dataset, val_split.indices)
        test_data = torch.utils.data.Subset(test_dataset, test_split.indices)

        # Define train and test loaders
        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers,
            pin_memory=True, prefetch_factor=4 if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
        val_loader = DataLoader(
            val_data, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            pin_memory=True, prefetch_factor=4 if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
        test_loader = DataLoader(
            test_data, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            pin_memory=True, prefetch_factor=4 if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
        logger.info(f"DataLoaders ready. Train: {train_size}, Validation: {val_size}, Test: {test_size}")
        return train_loader,val_loader,test_loader