# SMS Spam Prediction App

## Overview

Welcome to the SMS Spam Prediction App! This is a data science project built with Streamlit that uses machine learning to predict whether an SMS message is spam or not. It’s designed to help users filter out unwanted messages and improve communication efficiency.

Did you know that **47% of consumers stop purchasing from a brand after losing trust in its digital security**, and **84% would consider switching to another brand** if that trust is lost? [ref](https://www.digicert.com/news/digicert-survey-highlights-importance-of-digital-trust-in-business-outcomes-customer-loyalty) Issues like SMS spam or unauthorized messages can damage this trust and lead to higher customer churn. 
If you're an email or SMS marketer, this app can help you check whether the message you've designed to engage your customers could be considered spam before sending it out helping you protect your brand's reputation and customer trust.

## Problem It Solves

In the digital age, SMS spam can be a major nuisance and harm your business by:
- Damaging your brand's reputation
- Increasing regulatory and operational costs
- Reducing the effectiveness of marketing messages

This app provides a way to automatically classify SMS messages as spam or not, helping you:
- Maintain a positive brand image
- Comply with regulations
- Optimize your communication strategy

## Tech Stack

Here’s what i used to build the app:

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://a11ybadges.com/badge?logo=streamlit", width=150/>
  

</div>

## Data Used
- Data set i used is from kaggel named [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

  
## How to Run the App

### Prerequisites

Before running the app, make sure you have Python installed on your system. You also need to install the required libraries. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

### Running the App

1. **Clone the Repository**: First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/Aashwin-Kumar/SMS_Spam_Detector.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd SMS_Spam_Detector
   ```

3. **Start the Streamlit App**: Run the following command to start the app:

   ```bash
   streamlit run app.py
   ```

4. **Open the App**: Once the app is running, open your web browser and go to `http://localhost:8501` to see the app in action.

## Usage

- **Input SMS**: Enter an SMS message into the provided text box.
- **Predict**: Click the "Predict" button to see whether the message is classified as spam or not.
- **View Results**: The app will display the prediction result along with some relevant information.

## Contributing

If you’d like to contribute to this project, please fork the repository and submit a pull request with your changes. i welcome any improvements or suggestions!

## Acknowledgements

- Thanks to the Streamlit community for their support and resources.
- Special thanks to the authors of the machine learning algorithms used in this project.
- Special thanks to Nitish bhai!!


Thank You 😊❣️
