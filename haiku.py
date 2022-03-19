import torchvision.models as models
import torchvision.transforms as transforms
import json
from PIL import Image
import torch
import transformers
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

#libraries for generating poems
from transformers import pipeline, set_seed
import nltk
nltk.download("stopwords")
set_seed(42)
import random
import re
import syllapy
from nltk.corpus import stopwords
first_words = set(stopwords.words('english'))
first_words = list(first_words - set(("ain","am","an","and","aren","aren't","at","be","been","being","between","both","by","couldn","couldn't","d","doesn","doesn't","doing","don","don't","down","during","further","hadn","hadn't","hasn","hasn't","haven","haven't",\
                            "he","her","here","hers","herself","him","himself","i","isn","isn't","it","it's","its","itself","ll","m","ma","me","mightn","mightn't","mustn","mustn't","myself","needn","needn't","not","not","o","of","off","on","once",\
                            "or","other","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","shuld've","shouldn","shouldn't","t","than","that'll","theirs","them","themselves","there","these","they","those","through",\
                            "too","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","who","whom","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","yours","yourself","yourselves","didn","didn't","did","should've")))
from random_word import RandomWords
r = RandomWords()
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet



#libraries for using clip 
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
from IPython.display import Image as im
import os
from torchvision.datasets import CIFAR100
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

def gen_story(beginning_line,sub_first=False):
  ''' given the first few words, will generate a dramatic sentence. If sub_first = True, will return the sentence not containing the begininning_lines fed it'''
  story_gen = pipeline('text-generation', model="pranavpsv/gpt2-genre-story-generator")
  beginning_line = "<BOS> <drama>" + ' ' + beginning_line
  results = story_gen(beginning_line)
  for text in results:
    output = str(text.values())[14:-3].lower()
    if sub_first == True:
      output = re.sub(beginning_line.lower(),'',output)
    output = re.sub( "<bos> <drama>",'',output)
    output = re.sub('[\n]','',output) #removing \n
    output = re.sub(r"[^\w\s']",' ',output) #removing other punctuation
    output = re.sub(r'[\d]','', output) #removing numbers
    output = re.sub(' +',' ',output) #removing double spaces
       
  return output

def syllable_counter(sentence,max,last_line = False):
  '''function takes original sentence and returns sentence with max specified number of syllables'''
  count = 0
  line= ''
  #counts syllables in sentence and adds word to line if syllable is less than max, if syllable = max, then breaks
  for word in sentence.split():
    new_count = syllapy.count(word)
    if count + new_count <= max:
      count += new_count
      line = line + word + ' '
     
    else:
      break
 
  
  nouns = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
  
  #adding in "stop word" to increment syllable count by one if next word in actual sentence is more than one syllable
  if last_line == False: 
    while count < max:
      additional_words = ['life','need','heart']
      new_word = random.choice(additional_words)
      count += 1
      line += new_word
 

  
  else: #if last line = true, indicates we may need to end sentence with a noun if syllable count is short
  
    if count == (max - 2):
     #If we need 2 more syllabes, use the wordnet package to check for synonyms and antonyms for the last word that was over count. 
     synonyms = []
     antonyms = []
     confirmed = 0

    #This nested for loop creates lists of both synonyms and antonyms for the last word
     for syn in wordnet.synsets(word):
      for l in syn.lemmas():
       synonyms.append(l.name())
       if l.antonyms():
        antonyms.append(l.antonyms()[0].name())
       
      
      #While loop first checks if there are any synonyms or antonyms for the final word that are under two syllables. If so, it ends the loop. 
      while confirmed == 0:
        for i in synonyms:
          if syllapy.count(i)==2 and i in nouns:
             additional_noun = i
             line += additional_noun
             confirmed = confirmed+1
             break
        for i in antonyms:
          if syllapy.count(i)==2 and i in nouns:
             additional_noun = i
             line += additional_noun
             confirmed = confirmed+1
             break

        #After going through the synonyms and antonyms, it uses the random word package. It keeps generating nouns until there is one that meets the syllable requirement.
        r = RandomWords()
        b = r.get_random_word(hasDictionaryDef="true")
        if syllapy.count(b)==2 and b in nouns:
          additional_noun = b
          line += additional_noun
          confirmed = confirmed + 1
          

    elif count == (max - 3):

     synonyms = []
     antonyms = []
     confirmed = 0

     for syn in wordnet.synsets(word):
      for l in syn.lemmas():
       synonyms.append(l.name())
       if l.antonyms():
        antonyms.append(l.antonyms()[0].name())
       
     
      while confirmed == 0:
        for i in synonyms:
          if syllapy.count(i)==3 and i in nouns:
             additional_noun = i
             line += additional_noun
             confirmed = confirmed+1
             break
        for i in antonyms:
          if syllapy.count(i)==3 and i in nouns:
             additional_noun = i
             line += additional_noun
             confirmed = confirmed+1
             break
        r = RandomWords()
        b = r.get_random_word(hasDictionaryDef="true", includePartOfSpeech="noun")
        if syllapy.count(b)==3 and b in nouns:
          additional_noun = b
          line += additional_noun
          confirmed = confirmed + 1
     
  
    elif count == (max - 1):
     
     synonyms = []
     antonyms = []
     confirmed = 0

     for syn in wordnet.synsets(word):
      for l in syn.lemmas():
       synonyms.append(l.name())
       if l.antonyms():
        antonyms.append(l.antonyms()[0].name())
       
      #additional_noun = 'problem'
      while confirmed == 0:
        for i in synonyms:
          if syllapy.count(i)==1 and i in nouns:
             additional_noun = i
             line += additional_noun
             confirmed = confirmed+1
             break
        for i in antonyms:
          if syllapy.count(i)==1 and i in nouns:
             additional_noun = i
             line += additional_noun
             confirmed = confirmed+1
             break
        r = RandomWords()
        b = r.get_random_word(hasDictionaryDef="true", includePartOfSpeech="noun")
        if syllapy.count(b)==1 and b in nouns:
          additional_noun = b
          line += additional_noun
          confirmed = confirmed + 1

  
  return line

def gen_poem2(label):
  poem = []
  label = re.sub('_',' ',label)
  first_word = random.choice(first_words) + ' ' + label
  sentence = gen_story(first_word, False)

  first_line = syllable_counter(sentence,5)

  poem.append(first_line)

  poem.append('\n')

  second_line = gen_story(first_line, True)

  second_line = syllable_counter(second_line,7)
  poem.append(second_line)
  
  total = first_line + ' ' + second_line
  third_line = gen_story(total, True)
  third_line = syllable_counter(third_line,5,last_line = True)
  poem.append(third_line)

  haiku = ''
  for i in poem:
    haiku += i 
  haiku = haiku[:-1]
  

  return haiku
 


img = Image.open('imgs/cat.jpg').convert('RGB') #input image

vgg16 = models.vgg16(pretrained = True)

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_2 = normalize(transform(img))
img_2 = img_2.unsqueeze(0)
prediction = vgg16(img_2)
prediction.data.numpy().argmax()
#download laebls: 
labels = json.load(open('imagenet_class_index.json'))
input = labels[str(prediction.data.numpy().argmax())][1]
input = re.sub('_',' ',input)



list_haikus = [gen_poem2(input)]

#for i in range(10):
#  haiku = gen_poem2(input)
#  list_haikus.append(haiku)



label = []
percent = []

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Prepare the inputs
image = img
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in list_haikus]).to(device)


# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(1)

# Print the result
print("\nTop prediction:\n")
for value, index in zip(values, indices):
    print(f"{list_haikus[index]}: {100 * value.item():.2f}%")
    label.append(list_haikus[index])
    percent.append(value.item())