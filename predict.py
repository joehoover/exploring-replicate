# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List, Union
import torch

import os

from transformers import AutoTokenizer#, AutoModelForSequenceClassification, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline

MODEL_REPO = "facebook/bart-large-mnli"
LOCAL_MODEL_DIR = './model'
LOCAL_ONNX_MODEL_DIR = "./bart_large_mnli_onnx/"



class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
        self.model = ORTModelForSequenceClassification.from_pretrained(LOCAL_ONNX_MODEL_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

        self.classifier = pipeline("zero-shot-classification",
            model = self.model,
            tokenizer=self.tokenizer, 
            device=0 if self.device == 'cuda' else -1,
        )

    def predict(
        self,
        input: str = Input(
            description = "Text sequence to classify.",
            default = "Replicate, I think I might...like you a lot!",
        ),
        class_labels: str = Input(
            description = "Class names. Must be a comma-delimited string of labels.",
            default = "positive, negative, neutral",
        ),
        multi_label: bool = Input(
            description = "If True, then class scores are independent.",
            default = False
        ),
        hypothesis_template: str = Input(
            description = "Hypothesis into which class labels are piped. Must contain '{}'.",
            default = "This example is {}."
        )
    ) -> dict:
        """Run a single prediction on the model"""

        class_labels = [i.strip() for i in class_labels.split(',')]
        result = self.classifier(input, class_labels, hypothesis_template=hypothesis_template, multi_label=multi_label)
        result["hypothesis_template"] = hypothesis_template
        return result