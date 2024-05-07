# chatbot
ChatBot using Gemini API and streamlit and pdf download from the google drive link
Overview
This ChatBot is a conversational bot designed to address various questions. It automatically processes information from a provided PDF file and leverages Gemini API for processing techniques and respond to user inquiries effectively.
The download.py file contains the code for the google drive link download to specific folder 

## Installation To run this ChatBot, you need to follow these steps:
Ensure that the google Drive API is enable.
Add the data folder to current directory so that the file can be downloaded in that folder 
Ensure you have Python installed on your system.
Clone this repository to your local machine.
Install the required Python packages by running:
Set up your Google API key by creating a .env file and adding your key:
Usage
Once you have installed the necessary dependencies and set up your Google API key, you can run the application using Streamlit. Execute the following command:

Streamlit run chatbot.py
This command will start the Streamlit server, and you can access the application in your web browser.

Features
The PDF link should be provided in the text box and enter , then click the  download button  processng will take place and the chat bar will appear .
Automatic PDF Processing: The application automatically extracts text from a provided PDF file.
Conversational Interface: Users can interact with the bot by asking questions related to the content of the PDF file.
Dynamic Chat History: The application maintains a chat history, displaying both user questions and bot responses.
Natural Language Understanding: The bot employs advanced language models to understand and respond to user queries effectively.
Contributing
Contributions to this ChatBot project are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Limitation :
As this can be used only by the user whose credetiasl are given to make the google drive API .
