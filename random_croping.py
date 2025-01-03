import os,cv2,json
import numpy as np

class ResizeRandomCrop:
  def __init__(self,data_file="data",crop_size=(224,224),rescale_size=None,num_of_crops=3,annotations=None):
    self.data_file = data_file #dossier d'images
    self.crop_size = crop_size #taille du crop
    self.num_of_crops = num_of_crops #nombre de crops par image
    self.rescale_size=rescale_size #taille de redimensionnement
    self.image_paths = os.listdir(data_file) #liste des fichiers dans le dossier
    self.output_folder = "output" #dossier de sortie
    self.annotations = annotations #fichier d'annotations
    if os.path.exists(self.output_folder)==False:
      os.makedirs(self.output_folder)
    print("\n Object of class \"RandomCrop\" has been initialized.")

  def resize(self,image_path):
      #Lecture de l'image
      image = cv2.imread(os.path.join(self.data_file,image_path)) 
      #Redimensionnement de l'image
      image=cv2.resize(image,self.rescale_size)
      return image

  def crop(self):
    new_annotation={}
    #Lecture des images (et redimensionnement)
    for image_path in self.image_paths:
      if self.rescale_size is not None:
        image=self.resize(image_path)
      else:
        image = cv2.imread(os.path.join(self.data_file,image_path))
      h, w = image.shape[:2]

      #Création des coordonées de crops
      for i in range(self.num_of_crops):
        x = np.random.randint(0, w - self.crop_size[0])
        y = np.random.randint(0, h - self.crop_size[1])

        #Crop de l'image
        crop = image[y:y+self.crop_size[1], x:x+self.crop_size[0]]
        name=f"{image_path.split('.')[0]}_crop{i}"
        output_path = os.path.join(self.output_folder, f"{name}.jpg")

        #Sauvegarde de la nouvelle image crop
        cv2.imwrite(output_path, crop)

        #Si on a passé un fichier de label, on l'ouvre et on recupère les annotations
        if self.annotations is not None:
          with open(self.annotations, 'r') as f:
            labels = json.load(f)    
          old_h=labels[image_path.split('.')[0]]["height"]
          old_w=labels[image_path.split('.')[0]]["width"]
          #On crée un nouveau fichier d'annotations avec les images cropés et les nouveaux labels
          new_annotation[name] = {
                "file_name": f"{name}.jpg",
                "height": self.crop_size[1],
                "width": self.crop_size[0],
                "annotations": []
              }
          for box in labels[image_path.split('.')[0]]["annotations"]:
              x1,y1,width,height=box["bbox"]

              if self.rescale_size is not None:

                x1=self.rescale_size[0]*x1//old_w
                y1=self.rescale_size[1]*y1//old_h
                width=self.rescale_size[0]*width//w
                height=self.rescale_size[1]*height//h

              x2=x1+width//2 #Laisse une marge de 50% pour la detection
              y2=y1+height//2 
              category_id=box["category_id"]  
              #On vérifie que l'élement qui était sur l'image initial est bien dans la nouvelle image
              if x1>=x and x2<=x+self.crop_size[0] and y1>=y and y2<=y+self.crop_size[1]:

                new_x1=x1-x
                new_y1=y1-y
                new_bbox=[new_x1,new_y1,width,height]

                new_annotation[name]["annotations"].append({"category_id":category_id,"bbox":new_bbox})

      annotation_output_path = os.path.join(self.output_folder, "annotations_cropped.json")
      with open(annotation_output_path, 'w') as f:
        json.dump(new_annotation, f)



    print("\n Random cropping complete! \n\n Cropped images are stored in the \"output\" folder.")

cropper=ResizeRandomCrop(data_file="data/TestSample",crop_size=(1280,720),rescale_size=(1920,1080),num_of_crops=3,annotations="data/data_parsed.json")
cropper.crop()
