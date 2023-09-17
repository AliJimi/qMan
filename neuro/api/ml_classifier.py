# Let's kickstart this code party with some absolutely necessary imports!
# Who shows up to a party without friends? Not us!

import pandas as pd  # The spreadsheeter's dream
import numpy as np  # Number cruncher extraordinaire
from sklearn.ensemble import RandomForestClassifier  # Our decision-making buddy
from sklearn.model_selection import train_test_split  # The "you go here, you go there" for data
from sklearn.metrics import accuracy_score, classification_report  # The judgement panel
from sklearn.preprocessing import StandardScaler  # Because nobody likes an unfair game


# Load up that dataset like it's a plate at a buffet!
# Nom nom nom... data!
def get_classification_by_age(new_age):
    try:
        data_acc = np.load("data_acc.npy")
        data_age = np.load("data_age.npy")
        data_cpm = np.load("data_cpm.npy")
        data_sat = np.load("data_sat.npy")
        data_wpm = np.load("data_wpm.npy")
        print("Successfully loaded the data, you data-gobbler you!")
    except FileNotFoundError:
        print("Oopsie-doodle! Can't find the dataset files. Make sure it's in the right place, buddy.")
        return None

    # Train-Test Split! It's like sorting Skittles before eating them.
    X_train, X_test, y_train, y_test = train_test_split(
        [data_cpm, data_wpm, data_acc, data_age, data_sat], [], test_size=0.2, random_state=42
    )
    print("Data split successfully. It's like tearing up a dance floor!")

    # Let's normalize our features because nobody likes a show-off!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Time to call in the Forest Rangers, AKA our RandomForest Classifier!
    # It's like assembling a team of experts for a heist.
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Random Forest, ASSEMBLE!!!")

    # Train that model like you're training a Pokémon!
    rf_classifier.fit(X_train, y_train)
    print("Training complete. Our Random Forest is now a Random Jungle!")

    # Unleash the beast! Time for predictions!
    y_pred = rf_classifier.predict(X_test)
    print("Predictions made! Check 'em out!")

    # Drumroll, please! 🥁 Let's see how well we did!
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Function to predict the preferred model for a new user based only on age
    def predict_preferred_model(age):
        # Averaging out the features like sharing fries at a table
        avg_features = np.mean(X_train, axis=0)

        # New user data, filled with average values and the given age
        # Like making a smoothie with one special ingredient!
        new_user_data = np.array([avg_features[0], avg_features[1], avg_features[2], age, avg_features[4]]).reshape(1,
                                                                                                                    -1)

        # Time to let the Random Forest do its thing!
        predicted_model = rf_classifier.predict(scaler.transform(new_user_data))[0]

        return predicted_model

    predicted_model = predict_preferred_model(new_age)

    print(f"The Random Forest thinks a user of age {new_age} would groove with the {predicted_model} model!")
