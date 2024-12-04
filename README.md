IT Number - IT21379406
Name - P.D.T.M.Karunarathna

      Machine Learning MODEL To Identify The Language And Era Of Palm Leaf Manuscripts

Palm leaf manuscripts are a treasure trove of historical knowledge, often containing information about fields like medicine and religion. These manuscripts are predominantly written in Pali, Sanskrit, and Sinhala. Due to their fragile state and the complexity of their scripts, it is essential to identify their language and era for preservation and further research.

Machine Learning Model:
                 Algorithm Used:
                 Backpropagation Algorithm:A supervised learning method that calculates the gradient of the loss function (e.g., cross-entropy loss) with respect to each weight in the network.
                 Gradients are propagated backward through the layers using the chain rule of calculus.
                 Weights are updated based on the gradients to minimize the loss.

                 architecture Used:
                 Convolutional Neural Networks (CNNs)
                 
Dataset Details: For language identification, the dataset is labeled with Pali, Sinhala, and Sanskrit scripts.
                 For era classification, the dataset distinguishes between the Mahanuwara and Polonnaruwa eras.
                
Techniques Employed:
                 Data Preprocessing: Included image resizing, normalization, and feature extraction to ensure the dataset is ready for model training.
                 Augmentation: Used to artificially expand the dataset by applying transformations such as rotation, flipping, and scaling to make the model robust.
                 Hyperparameter Tuning: Optimization of learning rates, batch sizes, and epochs for improved accuracy.
                 Evaluation Metrics: Accuracy, precision, recall, and loss were monitored to evaluate the performance of the model.

Additional Algorithms:
                 If augmentation or preprocessing involved, techniques like OpenCV or TensorFlow/Keras utilities might have been used.
                 Softmax Classifier: Likely used in the final CNN layer for multi-class classification.

This approach ensures the effective identification of the script’s language and the historical era, aiding in the preservation and digitization of these invaluable manuscripts.


