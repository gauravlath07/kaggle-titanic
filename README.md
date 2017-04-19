# Kaggle Titanic Challenge

### Motivation
Machine Learning is a field which is growing rapidly. The aim for this project was to investigate and learn about different Machine Learning models and at the same time develop proficiency in Python ML tools like scikit-learn, NumPy and Pandas.

### Aim
Train a ML model and predict if a person aboard the titanic survived or not. 

### Result
Reached 90.3% accuracy in prediction after 12 tries.
Top 2% in the world.

### How I made it ?
The script investigates 6 parameters - 
* Sex
* Name
* Ticket Class
* Boarding Gate
* Age
* Number of siblings of the passenger
* The gate the passenger embarked from

First the data was cleaned using NumPy and Pandas. There were some missing fields for gate embarked which were filled by investigating the percentage's of the most entered gates.

### Feature Engineering
To improve the accuracy another parameter called "Name length" was added. It improved accuracy from 78% to 83.9% because people with longer names were found to be "more important" and had a better chance of surviving.
Another field added to improve prediction was title of the person which again improved accuracy by 3% because people with "important" titles like Sir and Countess were more likely to survive.

## Training Model
I started off with a logistics regression model but after a lot of testing and graphing I realized that the relation between data features and surviving wasn't linear so I instead moved to a Random Forest model with 8 decision trees which improved accuracy by 40%.
