import tensorflow as tf
import numpy as np
from PIL import Image
import io

class Model:

    def __init__(self):
        print("initializing")

        self.models= {}
        self.class_names = {}

        self.categories = {
            'fruit': ['apple fruit', 'banana fruit', 'cherry fruit', 'grapes fruit', 'mango fruit', 'strawberry fruit'],
            'pets' : ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'],
            'tools' : ['Gasoline Can', 'Gravel', 'Hammer', 'Rope', 'Screw Driver', 'Toolbox', 'Wrench', 'pliers'],
            'utensils' : ['apple fruit', 'banana fruit', 'cherry fruit', 'grapes fruit', 'mango fruit', 'strawberry fruit'],
        }

        self.load_models()

    def load_models(self):

        for category in self.categories.keys():
            #finds path to trained model
            model_path = f'../models/{category}_class.h5'

            try:
                #loads model through tf
                self.models[category] = tf.keras.model.load_model(model_path)
                self.class_names[category] = self.categroies[category]
                print(f"loaded {category} model")
            except:
                print(f"not loaded {category} model")
        
        print(f"loaded {len(self.models)} models")

