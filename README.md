# NER For Resume Entity Extraction Using XLMRoberta

This project focuses on Named Entity Recognition (NER) for resume entity extraction using the XLMRoberta transformer architecture. The goal is to extract entities such as name, email, skills, college, work experience, graduation year, companies worked at, and more (about 20 entities). The dataset is in a JSON format containing raw data, where each item includes text and entity information. The entity section includes labels and the position (start, end) of the entity in the text.

## Training

To train the model, follow these steps:

1. Install the required packages:

    ```bash
    pip install -r training_requirements.txt
    ```

2. Preprocess the data:

    ```bash
    python ./utils/preprocess-data.py
    ```

3. Tokenize the data:

    ```bash
    python ./utils/tokenize_data.py
    ```

4. Train the model:

    ```bash
    python ./utils/model.py --input data/input_data  --epochs 5 --lr 0.00005
    ```

In each stage, the data will be processed and saved in the `data` folder. You can specify the input and output directories using the `--input` and `--output` flags in the scripts. Additionally, in the `model.py` file, you can customize hyperparameters such as `--epochs`, `--lr`, and others.

## Inference

To test the trained model and use it for extraction, follow these steps:

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the inference script:

    ```bash
    python ./main.py
    ```

This will build a simple Gradio application to take your resume as a PDF and highlight each entity. The output screen will look like the screenshots below:

- ![Screenshot 1](https://github.com/MohamedSamy853/NER-For-Resume-Entity-Extraction-Using-XLMRoberta/blob/main/out.jpg)
- ![Screenshot 2](https://github.com/MohamedSamy853/NER-For-Resume-Entity-Extraction-Using-XLMRoberta/blob/main/out2.jpg)

Feel free to explore and enhance the functionality of the project!
