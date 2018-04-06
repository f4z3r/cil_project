#!/usr/bin/env python3

from models.abstract_model import AbstractModel
from models.road_sequence_generator import RoadSequenceGenerator


class AugmentationModel(AbstractModel):
    """A model template using extended Keras Augmentation for validation."""

    def create_train_batch(self, batch_size=100):
        return RoadSequenceGenerator(self.train_path, 'data', 'verify', batch_size)

    def create_validation_batch(self, batch_size=100):
        return RoadSequenceGenerator(self.validation_path, 'data', 'verify', batch_size)
