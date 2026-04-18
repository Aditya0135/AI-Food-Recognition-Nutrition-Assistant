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
 
    def get_test_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.config.resize_size),
            transforms.CenterCrop(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
 
    def load_and_split(self):
        train_transform = self.get_train_transform()
        test_transform = self.get_test_transform()

        dataset = datasets.ImageFolder(root=self.config.unzip_dir)

        logger.info(f"Classes found: {dataset.classes}")
        logger.info(f"Total images: {len(dataset)}")

        # define size
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        # get seed and train-test split
        seed_num = torch.Generator().manual_seed(self.config.seed)
        train_data, test_data = random_split(dataset,lengths=[train_size,test_size],generator=seed_num)

        # transform and augment
        train_data.dataset.transform = train_transform # type: ignore[attr-defined]
        test_data.dataset.transform = test_transform # type: ignore[attr-defined]

        # Define train and test loaders
        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers,
            pin_memory=True, prefetch_factor=4,
        )
        test_loader = DataLoader(
            test_data, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            pin_memory=True, prefetch_factor=4
        )
        logger.info(f"DataLoaders ready. Train: {train_size}, Test: {test_size}")
        return train_loader, test_loader