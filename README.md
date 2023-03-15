# exploring-replicate


## Note

I've modified `predict.py` to use onnx runtime; however, I have not gotten any version of my cog model to run on replicate.ai. So, the ONNX version is currently untested. However, it is working locally.

To export the model to ONNX:

```
pip install optimum[exporters]
optimum-cli export onnx --model model/ --task sequence-classification bart_large_mnli_onnx/
```

##

This spins up an instance of the HuggingFace Transformers `zero-shot-classification` pipeline with a large `BART` model that's been fine-tuned on NLI. It's an old approach (see [here](https://huggingface.co/facebook/bart-large-mnli?candidateLabels=mobile%2C+website%2C+billing%2C+account+access&multiClass=true&text=Last+week+I+upgraded+my+iOS+version+and+ever+since+then+my+phone+has+been+overheating+whenever+I+use+your+app.)) as far as zero shot inference is concerned, but it's also small and fast, compared to today's large models. 

The model formulates sequence classification as an NLI problem. Given a set of class labels, an input sequence, and a hypothesis template: 

1. A hypothesis is constructed for each label by piping labels into the hypothesis template.
2. Each hypothesis is appended to the user input, yielding the complete input sequence
3. The input sequence is passed to the NLI model, which then predicts whether the 
hypothesis contradicts, entails, or is irrelevant to the user input (i.e. the premise). 
4. Entailment logits associated with each hypothesis--remember, a hypothesis is constructed for each label--are then normalized with a softmax to yield the final scores over labels that are returned in the output.  

Outputs are returned as a dictionary with three keys: 

* `hypothesis_template`: The hypothesis template used to operationalize the full input sequence.
* `labels`: Class labels specified by the user ordered by `scores`
* `scores`: Scores associated with each label
* `sequence`: Input sequence used for classification 

# Run this locally...

1. Clone this repo

2. Install cog

```
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

3. `cd` into repo directory and run: 

```
cog predict -i input="This is so cool!"
```

**Note:** If you're using a lambda cloud gpu instance, you need to run docker (and cog -> docker) commands with `sudo`.

# Examples

Running this

```bash
cog predict -i input="Replicate, I think I might...like you a lot!"
```

returns

```
{
  "labels": [
    "positive",
    "neutral",
    "negative"
  ],
  "scores": [
    0.9709699153900146,
    0.00875716283917427,
    0.0003253005270380527
  ],
  "sequence": "Replicate, I think I might...like you a lot!"
}
```