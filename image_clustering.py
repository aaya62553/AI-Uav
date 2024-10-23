
import random, cv2, os, shutil
from sklearn.cluster import KMeans
import numpy as np



class image_clustering:

	def __init__(self, folder_path="data", n_clusters=5,max_examples=None):
		paths = os.listdir(folder_path)
		if max_examples == None:
			self.max_examples = len(paths)
		else:
			if max_examples > len(paths):
				self.max_examples = len(paths)
			else:
				self.max_examples = max_examples
		self.n_clusters = n_clusters
		self.folder_path = folder_path
		random.shuffle(paths)
		self.image_paths = paths[:self.max_examples]
		if os.path.exists("output")==False:
				os.makedirs("output")
		del paths 
		try:
			shutil.rmtree("output")
		except FileExistsError:
			pass
		os.makedirs("output")
		print("\n output folders created.")
		for i in range(self.n_clusters):
			os.makedirs("output/cluster" + str(i))
		print("\n Object of class \"image_clustering\" has been initialized.")

	def load_images(self):
		self.images = []
		for image in self.image_paths:
			image = cv2.imread(os.path.join(self.folder_path,image))
			image = cv2.resize(image, (224,224))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			self.images.append(image)

		self.images = np.float32(self.images).reshape(len(self.images), -1)
		self.images /= 255

		print("\n " + str(self.max_examples) + " images from the \"" + self.folder_path + "\" folder have been loaded in a random order.")



	def clustering(self):
		model = KMeans(n_clusters=self.n_clusters, random_state=728)
		model.fit(self.images)
		predictions = model.predict(self.images)

		#print(predictions)

		for i in range(self.max_examples):
			shutil.copy2(os.path.join(self.folder_path,self.image_paths[i]), "output/cluster"+str(predictions[i]))
		print("\n Clustering complete! \n\n Clusters and the respective images are stored in the \"output\" folder.")

if __name__ == "__main__":

	number_of_clusters = 2

	data_path = "val" 

	temp = image_clustering(data_path, number_of_clusters,100)
	temp.load_images()
	temp.clustering()

