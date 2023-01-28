import os
import sys
import torch
import clip
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


class ClipImage:
    def __init__(self, path_of_ppt_folders, path_to_save_image_features, mode='image', device='cuda'):
        """
        :param input_image_path: path of the input image (mode = 'image') or the actual text to be searched (mode='text')
        :param path_of_ppt_folders: path of the folder containing all the ppt folders
        :param path_to_save_image_features: path to save the image features
        :param mode: 'image' or 'text' based on the type of input
        :param device: device to run the model on
        """

        self.res = []
        if not os.path.isdir(path_of_ppt_folders):
            raise TypeError(
                f"{path_of_ppt_folders} is not a directory. Please only enter a directory")

        # if mode == 'image' and not os.path.exists(input_image_path):
        #     raise FileNotFoundError(f"{input_image_path} does not exist.")
        if not os.path.exists(path_to_save_image_features) or not os.path.isdir(path_to_save_image_features):
            raise FileNotFoundError(
                f"{path_to_save_image_features} is not a directory or doesn't exist.")
        self.mode = mode
        self.path_of_ppt_folders = path_of_ppt_folders
        self.path_to_save_image_features = path_to_save_image_features
        self.device = device
        
        # consider ViT-L/14 should be the best one
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        
        print("👉 RUNNING CLIP'S ONE-TIME ENCODING STEP... will be slow the first time, and hopefully only the first time.")
        # passing in an image as a cheap hack, to make one funciton work for initial embedding.
        self.calculate_similarity('/home/kastanday/uiuc_chatbot/main_fn/lecture_slides/001/Slide1.jpeg')
        print("🔥 DONE with CLIP's ONE TIME ENCODING")
    
    def text_to_image_search(self, search_text: str, top_k_to_return: int = 4):
        """ Written after the fact by kastan, so that we don't have to call init every time. """
        assert type(search_text) == str, f"Must provide a single string, instead I got type {type(search_text)}"
        # self.create_input_features(search_text, mode='text')
        self.mode = 'text'
        return self.calculate_similarity(search_text, top_k_to_return)

    # TODO: WIP.
    def image_to_images_search(self, input_image, top_k_to_return: int = 4):
        """ Written after the fact by kastan, so that we don't have to call init every time. """
        assert type(search_text) == str, f"Must provide a single string, instead I got type {type(search_text)}"
        # self.create_input_features(search_text, mode='text')
        self.mode = 'image'
        return self.calculate_similarity(input_image, top_k_to_return)
      

    def create_input_features(self, input_text_or_img):
        if self.mode == 'image':
            # Load the image
            input_image = Image.open(input_text_or_img)
            # Preprocess the image
            input_arr = torch.cat(
                [self.preprocess(input_image).unsqueeze(0)]).to(self.device)

        elif self.mode == 'text':
            # Preprocess the text
            input_arr = torch.cat(
                [clip.tokenize(f"{input_text_or_img}")]).to(self.device)

        # Encode the image or text
        with torch.no_grad():
            if self.mode == 'image':
                input_features = self.model.encode_image(input_arr)
            elif self.mode == 'text':
                input_features = self.model.encode_text(input_arr)
        input_features /= input_features.norm(dim=-1, keepdim=True)
        return input_features

    def most_similar_slide_file(self, top_k: int):
        # Sort the results
        ans = sorted(self.res, key=lambda x: x[2], reverse=True)
        return ans[:top_k]

    def calculate_similarity(self, input_text_or_img, topk_val: int = 3):
        # Create the input features
        input_features = self.create_input_features(input_text_or_img)
            
        # Iterate through all the folders
        ppts = list(os.listdir(self.path_of_ppt_folders))

        for i in ppts:
            # Get the path of the folder containing the ppt images
            imgs = list(os.listdir(os.path.join(self.path_of_ppt_folders, i)))
            # Iterate through all the images and preprocess them
            image_input = torch.cat([self.preprocess(Image.open(os.path.join(
                self.path_of_ppt_folders, i, image))).unsqueeze(0) for image in imgs]).to(self.device)

            # Check if the preprocessed file exists and load it
            img_flag = os.path.exists(
                self.path_to_save_image_features+i+"_tensor.pt")
            if img_flag:
                image_features = torch.load(
                    self.path_to_save_image_features+i+"_tensor.pt", map_location=self.device)
            else:
                # Encode the images and save the encoding
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                torch.save(image_features,
                           self.path_to_save_image_features+i+"_tensor.pt")
                print("Saved the image features (for faster future loading) to: ",
                      self.path_to_save_image_features+i+"_tensor.pt")

            # Calculate the similarity between the input image and the images in the folder
            
        # TODO: THIS REQUIRES REFACTOR. We're only looking in a SINGLE FOLDER. need to APPEND to similarity. 
            if self.mode == 'image':
                similarity = (100.0 * input_features @
                              image_features.T).softmax(dim=-1)
            elif self.mode == 'text':
                similarity = (100.0 * image_features @
                              input_features.T).softmax(dim=-1)

        # Get the top k most similar images
        print("🚨FIXING TOP K to 1!!!!!!!!!!!!!!")
        topk_val = 1
        print(similarity)
        values, indices = similarity[0].topk(topk_val)
        for val, index in zip(values, indices):
            self.res.append([i, imgs[index], val])

        # Return the sorted results
        return self.most_similar_slide_file(topk_val)
