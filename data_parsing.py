import json

def data_parser(data_path,name="data_parsed.json"):
  data_parsed={}
  #On ouvre le fichier json
  data=json.load(open(data_path))

  #On parcourt les images
  for image_data in data["images"]:

    #Pour chaque image, on recupère l'id, la hauteur, la largeur et le nom du fichier
    image_id=image_data["id"]
    height=image_data["height"]
    width=image_data["width"]
    file_name=image_data["file_name"]
    annotations_list = []

    #On parcourt les annotations
    for annotation in data["annotations"]:
        #Si l'id de l'annotation correspond à l'id de l'image, on recupère la catégorie et les coordonées du bbox
        if annotation["image_id"]==image_id:
            annotations_list.append({"category_id":annotation["category_id"],"bbox":annotation["bbox"]})
    #Si on a pas d'annotations, on ajoute la catégorie 0 avec un bbox nul
    if len(annotations_list)==0:
        annotations_list.append({"category_id":0,"bbox":[0,0,0,0]})
    #On ajoute les informations dans le dictionnaire
    data_parsed[image_id]={"file_name":file_name,"height":height,"width":width,"annotations":annotations_list}

  #On sauvegarde le dictionnaire dans un fichier json
  with open(name, 'w') as f:
    json.dump(data_parsed, f)

