import tensorflow as tf
import numpy as np
from PIL import Image
import io

class Model:

    def __init__(self):
        print("initializing")

        self.models= {}
        self.class_names = {}

        #load in categories identical to input of actual models
        self.categories = {
            'fruit': ['apple fruit', 'banana fruit', 'cherry fruit', 'grapes fruit', 'mango fruit', 'strawberry fruit'],
            'pets' : ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'],
            'tools' : ['Gasoline Can', 'Gravel', 'Hammer', 'Rope', 'Screw Driver', 'Toolbox', 'Wrench', 'pliers'],
            'utensils' : ['BOTTLE_OPENER', 'BREAD_KNIFE', 'CAN_OPENER', 'DESSERT_SPOON', 'DINNER_FORK', 'DINNER_KNIFE', 'FISH_SLICE', 'KITCHEN_KNIFE', 'LADLE', 'MASHER', 'PEELER', 'PIZZA_CUTTER', 'POTATO_PEELER', 'SERVING_SPOON', 'SOUP_SPOON', 'SPATULA', 'TEA_SPOON', 'TONGS', 'WHISK', 'WOODEN_SPOON'],
        }

        self.load_models()

    def load_models(self):

        for category in self.categories.keys():
            #finds path to trained model
            model_path = f'../models/{category}_class.h5'

            try:
                #loads model through tf
                self.models[category] = tf.keras.models.load_model(model_path)
                self.class_names[category] = self.categories[category]
                print(f"loaded {category} model")
            except Exception as e:
                print(f"not loaded {category} model {e}")
        
        print(f"loaded {len(self.models)} models")

    #processes image for the model, must be specific dimension and tones
    def image_processing(self, image_bytes):
        #opens image from raw bytes
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((256,256))        
        image_array = np.array(image)
        #between 0-1
        image_array = image_array/255.0

        image_array = np.expand_dims(image_array, axis = 0)

        #returns processed image
        return image_array

    #request gets inputted into here
    #main function
    def predict(self, image_bytes, category):
        
        if category not in self.models:
            return { 'success': False, 'error' : f'category {category} not found'}
        try: 
            #makes image fit the processing standards
            processed = self.image_processing(image_bytes)
            model = self.models[category]
            prediction = model.predict(processed, verbose = 0)

            #removes unnecessary batch dimensions, finds index with ighest number
            predict_perc = np.argmax(prediction[0])
            #percentage that it is the right value
            confidence = float(prediction[0][predict_perc])
            #class which mathces index with highest confidence
            pred_class = self.class_names[category][predict_perc]

            
            #hashmap of all probabilities predicted
            all_predict = {}
            for idx, class_name in enumerate(self.class_names[category]):
                all_predict[class_name] = float(prediction[0][idx])


            return {
                        'success': True,
                        'category': category,
                        'predicted_class': pred_class,
                        'confidence': confidence,
                        'confidence_percent': f"{confidence * 100:.1f}%", 
                        #'all_predictions': all_predict
                    }
        
        except Exception as e:
            return {
                    'success': False,
                    'error': str(e)
                }
