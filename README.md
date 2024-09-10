# Face_Recognition_app
so in this GitHub repository, we are going to be tackling the second part of the project which is, ****creating the face recognition app and launching it****, here you can check the code (if its correct and if its complete)

the repository contains ****the python code to create the app****, the two files created in the kaggle notebook ****the embeddings pickle file**** and ****the face index file**** and also a ****requirements file**** that contains all the necessary libraries for the python code. the two files created in kaggle were downloaded and uploaded into this repository.

### the python code:
the steps that the code conatins are:

1/loading the faiss index file and reading it: using faiss.read_index()

2/loading the embeddings from the embeddings pickle file: using pickle.load()

3/loading the InceptionResnetV1 embedding model: we need it to embed the face on the image loaded by the user later on

4/creating a function for preprocessing the images: (with the function created earlier on the kaggle notebook)

5/creating a function to embed the face on the loaded image: in this function the image is going to be preprocessed using the previous function, embedded using the embedding model and it will search for the closest match for the face

6/creating the app (using the stramlit library so that we are going to be able to launch the app into the streamlit share website)


****then go back to the kaggle notebook to be able to find the URL to the face recognition app which is ready to use :)****
