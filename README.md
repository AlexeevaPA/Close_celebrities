# Close_celebrities

![First_page](https://github.com/AlexeevaPA/Close_celebrities/assets/104028421/0a33c15d-3411-494b-878d-2a7f08ab5171)


**Overview**

This website aims to assist users in identifying the celebrity that bears the closest 
resemblance to them out of a selection of 17 celebrities.

**Data**

The training dataset utilized for model training was obtained from Kaggle and is referred to as 
the "Celebrity Face Image Dataset". To train the model, transfer learning was employed, with VGGFace serving as the base model.

**Framework**

The website is built using the Flask framework. The backend is developed using Python, 
while the frontend incorporates HTML, CSS, and a small amount of JavaScript code.

The website provides the functionality to upload a photo and view a list of the three 
celebrities who bear the closest resemblance to the user based on their appearance.

**Model**

The underlying structure of the prediction model is VGG16, with all layers except Flatten and Dense originating from VGGFace. 
To compile the model, the loss function "categorical_crossentropy" and the metric "accuracy" were employed. 
The final accuracy achieved by the model was 0.94.  


