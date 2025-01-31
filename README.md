TASK

1. To train an image classification model and deploy into an application

      DATASET PREPARATION : https://www.kaggle.com/datasets/ajaykgp12/cars-wagonr-swift
                            https://www.kaggle.com/datasets/tongpython/cat-and-dog
   
                            The dataset is labelled using roboflow (a tool which is used to annotate images).
   

      DATA AUGUMENTATION : The images in the dataset are reduced in size using train and test_Datagen methods. The height and width of the images are modified upto 20%, rotation range of 30 degree and some images are also flipped horizontally using ImageDataGenerator() method

      DEFINING THE LAYERS : Three convolutional layers and max pooling layers are used, they are with 32, 64 and 128 filters respectively. The activation function of the convolutional layers are Relu to make the model to learn accurately and 50% of the layers are dropped out to prevent overfitting and the final layer as a softmax layer as it return the probability of the prediction.

      MODEL TRAINING : To train the model the Adam optimizer is used, for an optimised training 20 epochs are defined to train the model.  The accuracy and the loss are calculated at each epoch and the final test_accuracy is displayed and the model is saved at the desired path.

      MODEL EVALUATION : To evaluate the model the trained model is loaded and the test image is given, the given image is converted into an array with the help of the array it detects which class the image belongs to and it displays the predicted class and confidence percentage of the image.

      DEPLOYMENT : A web application is developed using HTML (to test the model). Using flask to build server, the input image from the frontend is recieved and the image is provided to the model in the backend and the prediction by the model is sent to the frontend and the prediction is displayed in the frontend.
    
      
