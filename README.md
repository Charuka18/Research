IT Number - IT21379406
Name - P.D.T.M.Karunarathna
Project ID: 24-25J-013

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

Research question : How can machine learning models accurately identify the language and era of palm leaf manuscripts to enhance historical research and preservation efforts?

Solution Overview : 
                Data Collection and Preparation:
                        Collect a diverse dataset of palm leaf manuscripts in Pali, Sanskrit, and Sinhala languages.
                        Ensure the dataset includes samples from different historical periods, such as the Mahanuwara and Polonnaruwa eras.
                        Annotate the dataset with labels for language and era for supervised learning.
                        Perform preprocessing to handle image noise, normalize dimensions, and enhance readability.

                Feature Extraction:
                        Extract features from manuscript images, including textual patterns (e.g., letters per inch, character size) and structural patterns.
                        Utilize deep learning to automatically identify complex visual features from the scripts.
                        
                Model Development:
                        Use Convolutional Neural Networks (CNNs) to identify visual and textual patterns in manuscripts.
                        The CNN architecture is fine-tuned to classify languages (Pali, Sanskrit, Sinhala) and eras (Mahanuwara, Polonnaruwa).
                        
                Training and Optimization:
                        Train the CNN model using a labeled dataset, applying the backpropagation algorithm for weight optimization.
                        Employ techniques like dropout and batch normalization to avoid overfitting and improve generalization.
                        Use performance metrics (accuracy, precision, recall) to evaluate and refine the model.
                        
                Era and Language Identification:
                        The trained model outputs predictions on the language and era of new manuscript samples.
                        Predictions are based on the model's learned patterns and probabilities for each class.
                        
                Integration for Preservation and Research:
                        Implement the model in a user-friendly application for historians and researchers to upload manuscript images and receive classifications.
                        Provide visualizations and confidence levels for predictions, along with tools for further study.
                        
                Potential Expansion:
                        Expand the model to handle additional languages and historical periods by incorporating new datasets.
                        Integrate additional methods like Optical Character Recognition (OCR) for textual content extraction.
                
Techniques Employed:
                 Data Preprocessing: Included image resizing, normalization, and feature extraction to ensure the dataset is ready for model training.
                 Augmentation: Used to artificially expand the dataset by applying transformations such as rotation, flipping, and scaling to make the model robust.
                 Hyperparameter Tuning: Optimization of learning rates, batch sizes, and epochs for improved accuracy.
                 Evaluation Metrics: Accuracy, precision, recall, and loss were monitored to evaluate the performance of the model.

Additional Algorithms:
                 If augmentation or preprocessing involved, techniques like OpenCV or TensorFlow/Keras utilities might have been used.
                 Softmax Classifier: Likely used in the final CNN layer for multi-class classification.

This approach ensures the effective identification of the script’s language and the historical era, aiding in the preservation and digitization of these invaluable manuscripts.


