# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List, Union
import torch

import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# CACHE_DIR = 'weights/models--facebook--bart-large-mnli/snapshots/9fc9c4e1808b5613968646fa771fc43fb03995f2/'
CACHE_DIR = "./weights/models--facebook--bart-large-mnli/snapshots/9fc9c4e1808b5613968646fa771fc43fb03995f2/"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli", cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli", cache_dir=CACHE_DIR)
        self.model.to(self.device)


        self.classifier = pipeline("zero-shot-classification",
            model = self.model,
            tokenizer=self.tokenizer, 
            device=0 if self.device == 'cuda' else -1,
        )

    def predict(
        self,
        input: str = Input(
            description = "Text sequence to classify",
            default = "I ate the most delicious plum yesterday!",
        ),
        class_labels: str = Input(
            description = "Class names. Must be either a comma-delimited string or a list of strings",
            default = "positive, negative, neutral",
        ),
        multi_label: bool = Input(
            description = "If True, then class scores are independent.",
            default = True
        ),
        hypothesis_template: str = Input(
            description = "Hypothesis into which class labels are piped. Must contain '{}'.",
            default = "This example is {}."
        )
    ) -> dict:
        """Run a single prediction on the model"""

        class_labels = [i.strip() for i in class_labels.split(',')]
        result = self.classifier(input, class_labels, hypothesis_template=hypothesis_template, multi_label=multi_label)
        return result