import streamlit as st
import requests
import json
import random
import numpy as np
from PIL import Image

import torch 
# from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from streamlit_option_menu import option_menu
st.set_page_config(page_title='Paraphaser Tool', page_icon="🎀")

def main():
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}    
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    
    # Applying styles to the buttons
    st.markdown("""<style>
                        .st-eb {
                            background-color:#F9786F
                        } </style>""", unsafe_allow_html=True)

    selected2 = option_menu(None, ["Tool", "About Us", "Source Code"], 
    icons=['map', 'people', "body-text"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

    if selected2== "Tool":
        
        st.write(" ")
        st.write(" ")
        # Heading
        st.header("AI based Paraphrasing Tool ", anchor=None)

        # Text area for user input
        user_input = st.text_area(" ", "Enter you text here and hit Phrase It")
        fact= ['Our app works best with question based prompts', 'If a mosquito is biting you, and you flex that muscle, it will most likely explode.','Dr Anand Pandey is the coolest faculty in SRM', 'The word "oxymoron" is itself an example of an oxymoron, as it is made up of two contradictory words: "oxy" meaning sharp and "moron" meaning dull.', 'There is a species of jellyfish called the immortal jellyfish that can live forever.']
        sample_text = ["Which course should I take to get started in data science?", "What is the best possible approach to learn aeronautical engineering?", "Do apples taste better than oranges in general?", "What are the ingredients required to bake a perfect cake?"]

        # sentence = "Which course should I take to get started in data science?"
        # sentence = "What is the best possible approach to learn aeronautical engineering?"
        # sentence = "Do apples taste better than oranges in general?"
        # sentence = "What are the ingredients required to bake a perfect cake?"


        # Phrase it button
        # if st.button("Button 1"):
        col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)
        with col4:
            phrase_button = st.button("Phrase It", type="primary")

        if phrase_button:

            # Checking for exceptions
            # if not check_exceptions(decoding_params):

            # Calling the forward method on click of Phrase It
            ran = random.randrange(0,len(fact)-1, 1)
            with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
                output = forward(user_input)
        
        # st.divider()
        st.write(" ")

        st.subheader("_Sample Sentences_", help="Click any sample sentence to get its Paraphrase")
        button1 = st.button(sample_text[0])
        button2 = st.button(sample_text[1])
        # button3 = st.button(sample_text[2])
        button4 = st.button(sample_text[3])
        st.write(" ")

        if button1:
            user_input = sample_text[0]
            ran = random.randrange(0,len(fact)-1, 1)
            with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
                output = forward(user_input)
        elif button2:
            user_input = sample_text[1]
            ran = random.randrange(0,len(fact)-1, 1)
            with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
                output = forward(user_input)        
        
        # elif button3:
        #     user_input = sample_text[2]
        #     ran = random.randrange(0,len(fact)-1, 1)
        #     with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
        #         output = forward(user_input)

        elif button4:
            user_input = sample_text[3]
            ran = random.randrange(0,len(fact)-1, 1)
            with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
                output = forward(user_input)


    if selected2== "About Us":
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header("Esteemed Guide:   ")
        st.write(" ")
        st.write(" ")
        st.subheader("Dr Anand Pandey, Asst Professor, CSE Deptt, SRM - NCR")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dr. Anand Pandey is Assistant Professor, Department of CSE at the Delhi-NCR Campus of SRM Institute of Science and Technology. His research interests lie in the areas of Machine Learning, Data Science, Python, Tableau, Cloud Computing, MANET and Internet of Things. Dr. Pandey holds a Ph.D in Computer Science and Engineering from Mewar University, Chittorgarh in 2016 and an M.Tech in Computer Science and Engineering from Kurukshetra University, Kurukshetra in 2000.")
        with col2:
            image = Image.open('dr-anand-pandey.jpg')
            st.image(image)#, caption='Sunrise by the mountains')
        
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header("Team:   ")

        st.write(" ")
        st.write(" ")
        st.subheader("Qazi Shuaib Wasi")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write("Qazi is a diligent and creative individual with a keen eye towards design. He loves to dive deep in coding, development and current trends in AI. \nYou would find him spending evenings in parks, reading, and contemplating human existence.\n\n Connect with Qazi on [Linkedin](https://in.linkedin.com/in/qazi-shuaib-01)")
        with col2:
            image = Image.open('Qazi.jpeg')
            st.image(image)
        
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Akshat Pareek")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write("Akshat Pareek is an analytical senior B.Tech CSE student at SRM Institute of Science and Technology, who loves competitive programming and solving challenging algorithmic problems. Alongside his technical skills, he also possesses effective communication, and an unquenchable love for dogs.\n\n Connect with Akshat on [Linkedin](https://www.linkedin.com/in/akshat-pareek-779055204/)")
        with col2:
            image = Image.open('Akshat.jpeg')
            st.image(image)

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Zahid Hussain Khan")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write("Zahid being an intelligent and resourceful person, enjoys coding and problem solving, in general.  He is regarded as a hardworking and efficient student by both his peers and professors. In addition to his academic achievements, he is also known to be a kind-hearted and compassionate individual, always willing to help others in need.\n\n Connect with Zahid on [Linkedin](https://www.linkedin.com/in/zahid-khan-0527371a3)")
        with col2:
            image = Image.open('Zahid.jpg')
            st.image(image)

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Atulya")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write("Atulya is a dynamic and driven individual, possessing a natural creativity and self-motivation that fuels his passion for computer science. Atulya posses strong love for coding in Java and in enquiring about emerging trends in cloud computing. When he is not excelling in his academic pursuits, you can often find him enjoying quality time with his family.\n\n Connect with Atulya on [Linkedin](https://www.linkedin.com/in/atulya-598855205/)")
        with col2:
            image = Image.open('Atulya.jpeg')
            st.image(image)


    if selected2== "Source Code":
        st.write(" ")
        st.write(" ")
        st.write ("**Github Link:**"+"\t_https://github.com/pareekakshat/Major_Project_")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Code to Train")
        training_code = ''' import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="Quora_Paraphrasing_train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="Quora_Paraphrasing_val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))

args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=6,
    eval_batch_size=6,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

train_path = "paraphrase_data/Quora_Paraphrasing_train.csv"
val_path = "paraphrase_data/Quora_Paraphrasing_val.csv"

train = pd.read_csv(train_path)
print (train.head())

tokenizer = T5Tokenizer.from_pretrained('t5-base')


class ParaphraseDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=256):
        self.path = os.path.join(data_dir, type_path + '.csv')

        self.source_column = "question1"
        self.target_column = "question2"
        self.data = pd.read_csv(self.path)

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx, self.source_column], self.data.loc[idx, self.target_column]

            input_ = "paraphrase: "+ input_ + ' </s>'
            target = target + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


dataset = ParaphraseDataset(tokenizer, 'paraphrase_data', 'Quora_Paraphrasing_val', 256)
print("Val dataset: ",len(dataset))

data = dataset[61]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))

if not os.path.exists('t5_paraphrase'):
    os.makedirs('t5_paraphrase')

args_dict.update({'data_dir': 'paraphrase_data', 'output_dir': 't5_paraphrase', 'num_train_epochs':2,'max_seq_length':256})
args = argparse.Namespace(**args_dict)
print(args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

def get_dataset(tokenizer, type_path, args):
  return ParaphraseDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)



print ("Initialize model")
model = T5FineTuner(args)

trainer = pl.Trainer(**train_params)

print (" Training model")
trainer.fit(model)

print ("training finished")

print ("Saving model")
model.model.save_pretrained('t5_paraphrase')

print ("Saved model") '''


        st.code(training_code)
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Code to Run")
        running_code = '''
import streamlit as st
import requests
import json
import random
import numpy as np
from PIL import Image

import torch 
# from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from streamlit_option_menu import option_menu
st.set_page_config(page_title='Paraphaser Tool', page_icon="🎀")

def main():
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}    
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    
    # Applying styles to the buttons
    st.markdown("""<style>
                        .st-eb {
                            background-color:#F9786F
                        } </style>""", unsafe_allow_html=True)

    selected2 = option_menu(None, ["Tool", "About Us", "Source Code"], 
    icons=['map', 'people', "body-text"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

    if selected2== "Tool":
        
        st.write(" ")
        st.write(" ")
        # Heading
        st.header("AI based Paraphrasing Tool ", anchor=None)

        # Text area for user input
        user_input = st.text_area(" ", "Enter you text here and hit Phrase It")
        fact= ['Our app works best with question based prompts', 'Dr Anand Pandey is the coolest faculty in SRM', 'The word "oxymoron" is itself an example of an oxymoron, as it is made up of two contradictory words: "oxy" meaning sharp and "moron" meaning dull.', 'There is a species of jellyfish called the immortal jellyfish that can live forever.']
        sample_text = ["Which course should I take to get started in data science?", "What is the best possible approach to learn aeronautical engineering?", "Do apples taste better than oranges in general?", "What are the ingredients required to bake a perfect cake?"]

        # sentence = "Which course should I take to get started in data science?"
        # sentence = "What is the best possible approach to learn aeronautical engineering?"
        # sentence = "Do apples taste better than oranges in general?"
        # sentence = "What are the ingredients required to bake a perfect cake?"


        # Phrase it button
        # if st.button("Button 1"):
        col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)
        with col4:
            phrase_button = st.button("Phrase It", type="primary")

        if phrase_button:

            # Checking for exceptions
            # if not check_exceptions(decoding_params):

            # Calling the forward method on click of Phrase It
            ran = random.randrange(0,len(fact)-1, 1)
            with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
                output = forward(user_input)
        
        # st.divider()
        st.write(" ")

        st.subheader("_Sample Sentences_", help="Click any sample sentence to get its Paraphrase")
        button1 = st.button(sample_text[0])
        button2 = st.button(sample_text[1])
        # button3 = st.button(sample_text[2])
        button4 = st.button(sample_text[3])
        st.write(" ")

        if button1:
            user_input = sample_text[0]
            ran = random.randrange(0,len(fact)-1, 1)
            with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
                output = forward(user_input)
        elif button2:
            user_input = sample_text[1]
            ran = random.randrange(0,len(fact)-1, 1)
            with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
                output = forward(user_input)        
        
        # elif button3:
        #     user_input = sample_text[2]
        #     ran = random.randrange(0,len(fact)-1, 1)
        #     with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
        #         output = forward(user_input)

        elif button4:
            user_input = sample_text[3]
            ran = random.randrange(0,len(fact)-1, 1)
            with st.spinner('T5 is processing your text ...       \nFun fact: '+ fact[ran]):
                output = forward(user_input)


    if selected2== "About Us":
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header("Esteemed Guide:   ")
        st.write(" ")
        st.write(" ")
        st.subheader("Dr Anand Pandey, HOD [ I T ], SRM - NCR")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dr. Anand Pandey is the Head of the Department of Information Technology at the Delhi-NCR Campus of SRM Institute of Science and Technology. His research interests lie in the areas of Machine Learning, Data Science, Python, Tableau, Cloud Computing, MANET and Internet of Things. Dr. Pandey holds a Ph.D in Computer Science and Engineering from Mewar University, Chittorgarh in 2016 and an M.Tech in Computer Science and Engineering from Kurukshetra University, Kurukshetra in 2000. You can contact him at hod.it.ncr@srmist.edu.in.")
        with col2:
            image = Image.open('dr-anand-pandey.jpg')
            st.image(image)#, caption='Sunrise by the mountains')
        
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header("Team:   ")

        st.write(" ")
        st.write(" ")
        st.subheader("Qazi Shuaib Wasi")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write("Qazi is a diligent and creative individual with a keen eye towards design. He loves to dive deep in coding, development and current trends in AI. \nYou would find him spending evenings in parks, reading, and contemplating human existence.\n\n Connect with Qazi on [Linkedin](https://in.linkedin.com/in/qazi-shuaib-01)")
        with col2:
            image = Image.open('Qazi.jpeg')
            st.image(image)
        
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Akshat Pareek")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write("Akshat Pareek is an analytical senior B.Tech CSE student at SRM Institute of Science and Technology, who loves competitive programming and solving challenging algorithmic problems. Alongside his technical skills, he also possesses effective communication, and an unquenchable love for dogs.\n\n Connect with Akshat on [Linkedin](https://www.linkedin.com/in/akshat-pareek-779055204/)")
        with col2:
            image = Image.open('Akshat.jpeg')
            st.image(image)

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Zahid Hussain Khan")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write("Zahid being an intelligent and resourceful person, enjoys coding and problem solving, in general.  He is regarded as a hardworking and efficient student by both his peers and professors. In addition to his academic achievements, he is also known to be a kind-hearted and compassionate individual, always willing to help others in need.\n\n Connect with Zahid on [Linkedin](https://www.linkedin.com/in/zahid-khan-0527371a3)")
        with col2:
            image = Image.open('Zahid.jpg')
            st.image(image)

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Atulya")
        st.write(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write("Atulya is a dynamic and driven individual, possessing a natural creativity and self-motivation that fuels his passion for computer science. Atulya posses strong love for coding in Java and in enquiring about emerging trends in cloud computing. When he is not excelling in his academic pursuits, you can often find him enjoying quality time with his family.\n\n Connect with Atulya on [Linkedin](https://www.linkedin.com/in/atulya-598855205/)")
        with col2:
            image = Image.open('Atulya.jpeg')
            st.image(image)


    if selected2== "Source Code":
        st.write(" ")
        st.write(" ")
        st.write ("**Github Link:**")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Code to Train")
st.code(training_code)
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Code to Run")
        running_code = " "
        st.code(running_code)
        st.write("Line 1")
        st.write("Line 2")
        st.write("Line 3")
        

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def forward(user_inp):
    # Making the request to the backend

    set_seed(42)
    # model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
    # # model = BERT_CLASS.from_pretrained('C:/Users/ASUS/Downloads/pytorch_model.bin', cache_dir=None, from_tf=False, state_dict=None, *input, **kwargs)
    # # model = T5ForConditionalGeneration.from_pretrained('C:/Users/ASUS/Downloads/pytorch_model.bin', cache_dir=None, from_tf=False, state_dict=None)
    # tokenizer = T5Tokenizer.from_pretrained('t5-base')

    tokenizer = AutoTokenizer.from_pretrained("F:\Serious Shit\SRM acads\8th Sem\Major Project\Review - 3")
    model = AutoModelForSeq2SeqLM.from_pretrained("F:\Serious Shit\SRM acads\8th Sem\Major Project\Review - 3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print ("device ",device)
    model = model.to(device)

    # sentence = "Which course should I take to get started in data science?"
    # sentence = "What is the best possible approach to learn aeronautical engineering?"
    # sentence = "Do apples taste better than oranges in general?"
    # sentence = "What are the ingredients required to bake a perfect cake?"
    sentence = user_inp

    
    # tokenizer.save_pretrained("F:\Serious Shit\SRM acads\8th Sem\Major Project\Review - 3")
    # model.save_pretrained("F:\Serious Shit\SRM acads\8th Sem\Major Project\Review - 3")
 
    text =  "paraphrase: " + sentence + " </s>"


    max_len = 256

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=10
    )


    st.write("## Paraphrased Text :: ")
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    for i, final_output in enumerate(final_outputs):
        st.write(final_output)
        # print("{}: {}".format(i, final_output))


if __name__ == "__main__":
    main()

'''
        st.code(running_code)
        # st.write("Line 1")
        # st.write("Line 2")
        # st.write("Line 3")
        

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def forward(user_inp):
    # Making the request to the backend

    set_seed(42)
    # # model = BERT_CLASS.from_pretrained('C:/Users/ASUS/Downloads/pytorch_model.bin', cache_dir=None, from_tf=False, state_dict=None, *input, **kwargs)
    # # model = T5ForConditionalGeneration.from_pretrained('C:/Users/ASUS/Downloads/pytorch_model.bin', cache_dir=None, from_tf=False, state_dict=None)
    # tokenizer = T5Tokenizer.from_pretrained('t5-base')

    tokenizer = AutoTokenizer.from_pretrained("F:\Serious Shit\SRM acads\8th Sem\Major Project\Review - 3")
    model = AutoModelForSeq2SeqLM.from_pretrained("F:\Serious Shit\SRM acads\8th Sem\Major Project\Review - 3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print ("device ",device)
    model = model.to(device)

    # sentence = "Which course should I take to get started in data science?"
    # sentence = "What is the best possible approach to learn aeronautical engineering?"
    # sentence = "Do apples taste better than oranges in general?"
    # sentence = "What are the ingredients required to bake a perfect cake?"
    sentence = user_inp

    
    # tokenizer.save_pretrained("F:\Serious Shit\SRM acads\8th Sem\Major Project\Review - 3")
    # model.save_pretrained("F:\Serious Shit\SRM acads\8th Sem\Major Project\Review - 3")
 
    text =  "paraphrase: " + sentence + " </s>"


    max_len = 256

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=10
    )


    st.write("## Paraphrased Text :: ")
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    for i, final_output in enumerate(final_outputs):
        st.write(final_output)
        # print("{}: {}".format(i, final_output))


if __name__ == "__main__":
    main()
